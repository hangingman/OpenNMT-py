""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn


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

        RNNCell = nn.LSTMCell if opt.lm_rnn_type == "LSTM"\
            else nn.GRUCell
        if opt.lm_rnn_type == "GRU":
            self.use_gru = True
        else:
            self.use_gru = False

        self.rnns = nn.ModuleList()
        if self.use_projection:
            self.projections = nn.ModuleList()

        for direction in range(self.num_directions):
            self.rnns.append(nn.ModuleList())
            if self.use_projection:
                self.projections.append(nn.ModuleList())

            rnn_input_size = self.input_size
            for layer in range(self.layers):
                rnn = RNNCell(rnn_input_size, self.hidden_size)
                self.rnns[direction].append(rnn)

                rnn_input_size = self.hidden_size
                if self.use_projection:
                    self.projections[direction].append(nn.Linear(
                                self.hidden_size,
                                self.input_size))

                    rnn_input_size = self.input_size

        self.gal_dropout = opt.lm_gal_dropout

        self.hidden = None

    def sample_mask(self, batch_size):
        """Create a mask for recurrent dropout
        Arguments:
            batch_size(int) -- the size of the current batch
        """
        keep = 1.0 - self.gal_dropout
        self.mask = Variable(torch.bernoulli(
                        torch.Tensor(batch_size,
                                     self.hidden_size).fill_(keep)))
        if self.is_cuda:
            self.mask = self.mask.cuda()

    def init_rnn_state(self, batch_size):
        """Initialize RNN state.
        Arguments:
            batch_size(int) -- the size of the current batch
        Returns:
            :obj:`Variable` -- the initial states of the LM RNNs
        """
        def get_variable():
            v = torch.zeros(self.num_directions, self.layers,
                            batch_size, self.hidden_size)
            if self.is_cuda:
                v = v.cuda()
            return v

        if self.rnn_type == 'LSTM':
            state = (get_variable(), get_variable())
        elif self.rnn_type == 'GRU':
            state = (get_variable(),)
        else:
            raise NotImplementedError("Not valid rnn_type: %s" % self.rnn_type)

        return state

    def forward(self, tgt, init_hidden):
        """Forward pass of the Language Model
        Arguments:
            tgt (:obj:`LongTensor`):
                 a sequence of size `[tgt_len x batch]` or
                 of size `[tgt_len x batch x num_characters]`
                 if we are using a character embedding model
            init_hidden(:obj:`Variable`) -- the initial states of the LM RNNs
        Returns:
            dir_outputs (Variable): the outputs from every RNN layer of the LM
            emb (Variable): the embeddings of the LM
        """

        # Get a dropout mask for recurrent dropout that will
        # be used for every timestep
        if self.gal_dropout > 0 and self.training:
            self.sample_mask(tgt.size(1))

        emb = self.embeddings(tgt)

        if self.num_directions > 1:
            reverse_emb = self._get_reverse_seq(tgt, emb)
            emb.unsqueeze_(0)
            reverse_emb.unsqueeze_(0)
            emb = torch.cat([emb, reverse_emb], dim=0)
        else:
            emb.unsqueeze_(0)

        dir_outputs = []
        output_cache = None

        for n_dir in range(self.num_directions):

            outputs = []
            h_0 = init_hidden[0][n_dir]
            if not self.use_gru:
                c_0 = init_hidden[1][n_dir]

            for i, emb_t in enumerate(emb[n_dir].split(1)):
                rnn_input = emb_t.squeeze(0)

                # Don't update if it is the first timestep
                if i > 0:
                    h_0 = h_1
                    if not self.use_gru:
                        c_0 = c_1

                h_1 = []
                if not self.use_gru:
                    c_1 = []

                layer_outputs = []

                for i, layer in enumerate(self.rnns[n_dir]):

                    if self.use_gru:
                        h_1_i = layer(rnn_input, h_0[i])
                    else:
                        h_1_i, c_1_i = layer(rnn_input, (h_0[i], c_0[i]))

                    output = h_1_i

                    # Apply recurrent dropout
                    if self.gal_dropout > 0 and self.training:
                        h_1_i = torch.mul(h_1_i, self.mask)
                        h_1_i *= 1.0/(1.0 - self.gal_dropout)

                    # Update hidden and cell state of this layer
                    h_1 += [h_1_i]
                    if not self.use_gru:
                        c_1 += [c_1_i]

                    # Project into smaller space
                    if self.use_projection:
                        rnn_input = self.projections[n_dir][i](output)
                    else:
                        rnn_input = output

                    # Residual Connection
                    if i > 0 and self.use_residual:
                        rnn_input.data += output_cache.data
                        output_cache = rnn_input
                    # Cache  the first layer output
                    elif self.use_residual:
                        output_cache = rnn_input

                    layer_outputs += [rnn_input]

                h_1 = torch.stack(h_1)
                if not self.use_gru:
                    c_1 = torch.stack(c_1)

                layer_outputs = torch.stack(layer_outputs)
                outputs += [layer_outputs]

            outputs = torch.stack(outputs)
            dir_outputs += [outputs]

        dir_outputs = torch.stack(dir_outputs)

        return dir_outputs, emb

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
