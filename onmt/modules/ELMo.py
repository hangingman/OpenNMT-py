import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.io


class ELMo(nn.Module):
    "An Implementation of ..."
    def __init__(self, language_model):
        super(ELMo, self).__init__()
        self.lang_model = language_model
        self.pad_idx = self.lang_model.padding_idx

        # Remove the language model parameters from the parameters
        # to be optimized
        for param in self.lang_model.parameters():
            param.requires_grad = False

        # Start with 1 parameter - the embeddings of the language model
        n_parameters = 1
        num_directions = len(self.lang_model.rnns)

        for direction in range(num_directions):
            n_parameters += len(self.lang_model.rnns[direction])

        self.softmax = nn.Softmax(dim=0)
        # Initialize to 0. - in the first pass the softmax will
        # distribute the weights uniformly this way
        layer_params = [nn.Parameter(torch.FloatTensor([0.0]))
                        for _ in range(n_parameters)]
        self.scalar_parameters = nn.ParameterList(layer_params)
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, tgt):

        _, batch_size, _, _ = tgt.size()

        # Initialize the hidden state and run the forward pass of
        # the language model
        init_hidden = self.lang_model.init_rnn_state(batch_size)
        outputs, emb = self.lang_model(tgt, init_hidden)

        num_directions, _, n_layers, _, _ = outputs.size()

        # Mask of the actual sequences without padding
        mask = ((tgt[:, :, 0, :] != self.pad_idx).sum(-1) > 0).long()

        # Remove the output embeddings that correspond to eos and bos tokens.
        no_bos_eos_emb, _ = self._remove_bos_eos_tokens(emb[0],
                                                        mask)

        # The embeddings are one of the layers of ELMo
        token_layers = [no_bos_eos_emb]

        # This loop is to process the layers of the output of the language
        # model in order to make them in the correct size and shape for
        # the ELMo computations.
        for direction in range(num_directions):

            dir_output = outputs[direction]

            for layer in range(n_layers):

                layer_output = dir_output[:, layer]
                # Remove the outputs that correspond to eos and bos tokens
                layer_output, _ = self._remove_bos_eos_tokens(layer_output,
                                                              mask)
                # Reverse the output of the backwards LM
                if direction != 0:
                    no_bos_eos_tensor, _ = self._remove_bos_eos_tokens(
                                tgt,
                                mask)
                    layer_output = self.lang_model._get_reverse_emb(
                                no_bos_eos_tensor, layer_output)

                token_layers.append(layer_output)

        # Get the normalized weights for each layer of the LM
        normed_weights = self.softmax(torch.cat([parameter for parameter
                                                 in self.scalar_parameters]))
        # Multiply weights with layers and sum everything
        pieces = []
        for weight, tensor in zip(normed_weights.split(1), token_layers):
            pieces.append(weight * tensor)

        return self.gamma * sum(pieces)

    def _remove_bos_eos_tokens(self, tensor, mask):

        self.pad_idx = self.lang_model.padding_idx
        sequence_lens = mask.sum(0)
        old_tensor_dims = list(tensor.size())
        new_tensor_dims = old_tensor_dims
        new_tensor_dims[0] = old_tensor_dims[0] - 2

        tensor_without_bos_eos = Variable(
                        tensor.data.new(*new_tensor_dims).fill_(self.pad_idx))
        new_mask = Variable(
                tensor.data.new(new_tensor_dims[0],
                                new_tensor_dims[1]).fill_(0)).long()

        for jj, ii in enumerate(sequence_lens):
            if int(ii) > 2:
                tensor_without_bos_eos[:int(ii-2), jj, :] = tensor[1:int(ii-1),
                                                                   jj, :]
                new_mask[:int(ii-2), jj] = 1

        return tensor_without_bos_eos, new_mask
