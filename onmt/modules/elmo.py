import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.io


class ELMo(nn.Module):
    "An Implementation of ..."
    def __init__(self, language_model, dropout, padding_idx):
        super(ELMo, self).__init__()
        self.lang_model = language_model.eval()
        self.pad_idx = padding_idx

        # Remove the language model parameters from the parameters
        # to be optimized
        for param in self.lang_model.parameters():
            param.requires_grad = False

        # Start with 1 parameter - the embeddings of the language model
        n_parameters = 1
        n_parameters += len(self.lang_model.forward_rnns)
        n_parameters += len(self.lang_model.backward_rnns)

        self.softmax = nn.Softmax(dim=0)
        # Initialize to 0. - in the first pass the softmax will
        # distribute the weights uniformly this way
        layer_params = [nn.Parameter(torch.FloatTensor([0.0]))
                        for _ in range(n_parameters)]
        self.scalar_parameters = nn.ParameterList(layer_params)
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))

        self.dropout = nn.Dropout(dropout)

    def forward(self, char_input):
        seq_len, batch_size, _, _ = char_input.size()

        # Mask of the actual sequences without padding
        mask = char_input[:, :, 0, :].ne(self.pad_idx).sum(-1)
        lengths = mask.sum(dim=0)

        # Set lang_model.training to false so the LM does not
        # use the custom dropout
        self.lang_model.dropout.training = False
        with torch.no_grad():
            outputs, _ = self.lang_model(char_input, lengths)

        outputs = outputs.view(*outputs.shape[:-1],
                               self.lang_model.num_directions,
                               -1)

        emb = outputs[0, :, :, 0, :]
        # Remove the output embeddings that correspond to eos and bos tokens.
        no_bos_eos_emb, _ = self._remove_bos_eos_tokens(emb,
                                                        mask)
        # The embeddings are one of the layers of ELMo
        bilm_layers = [no_bos_eos_emb]

        # This loop is to process the layers of the output of the language
        # model in order to make them in the correct size and shape for
        # the ELMo computations.
        for output in outputs[1:].split(1):

            forward_layer_output = output[0, :, :, 0, :]
            no_bos_eos_fl, _ = self._remove_bos_eos_tokens(
                forward_layer_output,
                mask)
            bilm_layers.append(no_bos_eos_fl)

            backward_layer_output = output[0, :, :, 1, :]
            no_bos_eos_bl, _ = self._remove_bos_eos_tokens(
                backward_layer_output,
                mask)
            bilm_layers.append(no_bos_eos_bl)

        # Get the normalized weights for each layer of the LM
        normed_weights = self.softmax(torch.cat([parameter for parameter
                                                 in self.scalar_parameters]))
        # Multiply weights with layers and sum everything
        pieces = []
        for weight, tensor in zip(normed_weights.split(1), bilm_layers):
            pieces.append(weight * tensor)

        return self.dropout(self.gamma * sum(pieces))

    def _remove_bos_eos_tokens(self, tensor, mask):

        sequence_lens = mask.sum(0)
        old_tensor_dims = list(tensor.size())
        new_tensor_dims = old_tensor_dims
        new_tensor_dims[0] = old_tensor_dims[0] - 2

        tensor_without_bos_eos = tensor.new_zeros(*new_tensor_dims)
        new_mask = tensor.new_zeros(
            *new_tensor_dims[:2], dtype=torch.long)

        for jj, ii in enumerate(sequence_lens):
            if int(ii) > 2:
                tensor_without_bos_eos[:int(ii-2), jj, :] = tensor[1:int(ii-1),
                                                                   jj, :]
                new_mask[:int(ii-2), jj] = 1

        return tensor_without_bos_eos, new_mask
