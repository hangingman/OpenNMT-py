""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn

from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class LanguageModel(nn.Module):
    """
    Trainable Language Model object.
    This implementation is heavily inspired in the models
    described in the following papers:
    - https://arxiv.org/pdf/1802.05365.pdf
    - https://arxiv.org/pdf/1602.02410.pdf
    - https://arxiv.org/abs/1608.05859.pdf
    Several options can be used:
    - Embedding and generator weight tying
    - Gal Dropout of hidden states (https://arxiv.org/pdf/1512.05287.pdf)
    - Residual/Skip Connections between RNN layers
    - Bidirectional LM option
    - Projection of RNN output into the same space of the embeddings
    - Possibility to use embeddings formed out of character convolutions

    Args:
      embeddings (:obj:`Embeddings` or
                  `CharEmbeddingsCNN`): word or character embeddings
      gpu (bool): if gpu is being used
      padding idx (int): the index of the padding symbol
    """
    def __init__(self, opt, embeddings, gpu, padding_idx):

        super(LanguageModel, self).__init__()

        self.is_cuda = True if gpu else False

        self.input_size = opt.lm_word_vec_size
        self.layers = opt.lm_layers
        self.embeddings = embeddings
        self.rnn_type = opt.lm_rnn_type
        self.hidden_size = opt.lm_rnn_size
        self.use_projection = opt.lm_use_projection
        self.use_residual = opt.lm_use_residual
        self.padding_idx = padding_idx
        self.num_directions = 2 if opt.lm_use_bidir else 1
        self.char_embeddings = opt.use_char_input

        RNN = getattr(nn, self.rnn_type)

        # Initialize RNNs
        self.forward_rnns = nn.ModuleList()
        if self.num_directions > 1:
            self.backward_rnns = nn.ModuleList()

        # Projections of the rnn output back to the input size
        if self.use_projection:
            self.forward_rnn_projections = nn.ModuleList()

            if self.num_directions > 1:
                self.backward_rnn_projections = nn.ModuleList()

        # RNN input size depends if we use projections or not
        rnn_input_size = self.input_size

        for layer in range(self.layers):
            # Initialize a forward rnn
            forward_rnn = \
                RNN(input_size=rnn_input_size,
                    hidden_size=self.hidden_size)

            self.forward_rnns.append(forward_rnn)

            if self.num_directions > 1:
                # Initialize a backward rnn
                backward_rnn = \
                    RNN(input_size=rnn_input_size,
                        hidden_size=self.hidden_size)

                self.backward_rnns.append(backward_rnn)

            # RNN input size is now the hidden size (output size),
            # unless we use projections
            rnn_input_size = self.hidden_size

            if self.use_projection:
                # Initialize projections to input space
                self.forward_rnn_projections.append(nn.Linear(
                                self.hidden_size,
                                self.input_size))

                if self.num_directions > 1:
                    self.backward_rnn_projections.append(nn.Linear(
                                    self.hidden_size,
                                    self.input_size))

                rnn_input_size = self.input_size

    def forward(self, tgt, lengths=None, hidden_state=None):
        """Forward pass of the Language Model
        Arguments:
            tgt (:obj:`LongTensor`):
                 a sequence of size `[tgt_len x batch]` or
                 of size `[tgt_len x batch x num_characters]`
                 if we are using a character embedding model
            hidden_state(:obj:`Variable`) -- the current states of the LM RNNs
        Returns:
            dir_outputs (Variable): the outputs from every RNN layer of the LM
            emb (Variable): the embeddings of the LM
        """

        # Go through the embedding layer
        forward_emb = self.embeddings(tgt)

        # Reverse the embeddings in the case of bidirectionality
        if self.num_directions > 1:
            backward_emb = self._get_reverse_seq(tgt, forward_emb)

        # Pack the embeddings if we know the lengths
        forward_input = forward_emb
        if lengths is not None:
            # Lengths data is wrapped inside a Tensor.
            lengths = lengths.view(-1).tolist()
            forward_input = pack(forward_emb, lengths)
            if self.num_directions > 1:
                backward_input = pack(backward_emb, lengths)

        layers_outputs = []
        forward_final_states = []
        backward_final_states = []

        forward_output_cache = None
        backward_output_cache = None

        if hidden_state is not None:
            forward_hidden_state = hidden_state[0]
            if self.num_directions > 1:
                backward_hidden_state = hidden_state[1]
        else:
            forward_hidden_state = [None for layer in range(self.layers)]
            backward_hidden_state = [None for layer in range(self.layers)]

        for layer in range(self.layers):

            forward_outputs, forward_layer_final_state = \
                        self.forward_rnns[layer](forward_input,
                                                 forward_hidden_state[layer])

            forward_outputs, _ = unpack(forward_outputs)

            if self.num_directions > 1:
                backward_outputs, backward_layer_final_state = \
                        self.backward_rnns[layer](backward_input,
                                                  backward_hidden_state[layer])

                backward_outputs, _ = unpack(backward_outputs)

            if self.use_projection:
                forward_input = self.forward_rnn_projections[
                                    layer](
                                    forward_outputs)

                if self.num_directions > 1:
                    backward_input = self.backward_rnn_projections[
                                    layer](
                                    backward_outputs)
            else:
                forward_input = forward_outputs
                if self.num_directions > 1:
                    backward_input = backward_outputs

            # Residual Connection
            if layer > 0 and self.use_residual:
                forward_input += forward_output_cache
                forward_output_cache = forward_input
                if self.num_directions > 1:
                    backward_input += backward_output_cache
                    backward_output_cache = backward_input
            # Cache  the first layer output
            elif self.use_residual:
                forward_output_cache = forward_input
                if self.num_directions > 1:
                    backward_output_cache = backward_input

            layers_outputs.append(forward_input)
            if self.num_directions > 1:
                layers_outputs.append(backward_input)

            forward_final_states.append(forward_layer_final_state)
            if self.num_directions > 1:
                backward_final_states.append(backward_layer_final_state)

            if lengths is not None:
                forward_input = pack(forward_input, lengths)
                if self.num_directions > 1:
                    backward_input = pack(backward_input, lengths)

        layers_outputs = torch.stack(layers_outputs)
        final_states = [forward_final_states,
                        backward_final_states]

        return forward_emb, layers_outputs, final_states

    def _get_reverse_seq(self, tgt, seq):
        """Get the reversed sequence (used for backward LM).
        """

        if self.char_embeddings:
            # Get the first character of each word.
            # If the word is actually padding, the first character will
            # already be the character padding token.
            tgt = tgt[:, :, 0]

        lengths = tgt.ne(self.padding_idx).sum(dim=0)[:, 0]
        sentence_size = int(lengths[0])
        batch_size = len(lengths)

        idx = [0]*(batch_size*sentence_size)

        for i in range(batch_size*sentence_size):
            batch_index = i % batch_size
            sentence_index = i//batch_size
            idx[i] = (int(lengths[batch_index])-sentence_index-1)*batch_size \
                + batch_index
            if idx[i] < 0:  # Padding symbol, don't change order
                idx[i] = i

        reversed_seq = seq.view(
                        batch_size*seq.size(0),
                        -1)[idx, :].view(seq.size(0),
                                         batch_size,
                                         -1)

        return reversed_seq
