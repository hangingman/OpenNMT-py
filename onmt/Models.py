import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import onmt
import onmt.modules
from onmt.modules import aeq
from onmt.modules.Gate import ContextGateFactory
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import math
import numpy as np
import pdb
import evaluation

class Embeddings(nn.Module):
    def __init__(self, opt, dicts, feature_dicts=None):
        self.positional_encoding = opt.position_encoding
        if self.positional_encoding:
            self.pe = self.make_positional_encodings(opt.word_vec_size, 5000) \
                          .cuda()

        self.word_vec_size = opt.word_vec_size

        super(Embeddings, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        # Word embeddings.
        self.dropout = nn.Dropout(p=opt.dropout)
        self.feature_dicts = feature_dicts
        # Feature embeddings.
        if self.feature_dicts is not None:
            self.feature_luts = nn.ModuleList([
                nn.Embedding(feature_dict.size(),
                             opt.feature_vec_size,
                             padding_idx=onmt.Constants.PAD)
                for feature_dict in feature_dicts])

            # MLP on features and words.
            self.activation = nn.ReLU()
            self.linear = onmt.modules.BottleLinear(
                opt.word_vec_size +
                len(feature_dicts) * opt.feature_vec_size,
                opt.word_vec_size)
        else:
            self.feature_luts = nn.ModuleList([])

    def make_positional_encodings(self, dim, max_len):
        pe = torch.FloatTensor(max_len, 1, dim).fill_(0)
        for i in range(dim):
            for j in range(max_len):
                k = float(j) / (10000.0 ** (2.0*i / float(dim)))
                pe[j, 0, i] = math.cos(k) if i % 2 == 1 else math.sin(k)
        return pe

    def load_pretrained_vectors(self, emb_file):
        if emb_file is not None:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, src_input):
        """
        Embed the words or utilize features and MLP.

        Args:
            src_input (LongTensor): len x batch x nfeat

        Return:
            emb (FloatTensor): len x batch x input_size
        """
        word = self.word_lut(src_input[:, :, 0])
        emb = word
        if self.feature_dicts is not None:
            features = [feature_lut(src_input[:, :, j+1])
                        for j, feature_lut in enumerate(self.feature_luts)]

            # Apply one MLP layer.
            emb = self.activation(
                self.linear(torch.cat([word] + features, -1)))

        if self.positional_encoding:
            emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                                 .expand_as(emb))
            emb = self.dropout(emb)
        return emb


class Encoder(nn.Module):
    """
    Encoder recurrent neural network.
    """
    def __init__(self, opt, dicts, feature_dicts=None):
        """
        Args:
            opt: Model options.
            dicts (`Dict`): The src dictionary
            features_dicts (`[Dict]`): List of src feature dictionaries.
        """
        # Number of rnn layers.
        self.layers = opt.layers

        # Use a bidirectional model.
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0

        # Size of the encoder RNN.
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts, feature_dicts)

        # The Encoder RNN.
        self.encoder_layer = opt.encoder_layer

        if self.encoder_layer == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerEncoder(self.hidden_size, opt)
                 for i in range(opt.layers)])
        else:
            self.rnn = getattr(nn, opt.rnn_type)(
                 input_size, self.hidden_size,
                 num_layers=opt.layers,
                 dropout=opt.dropout,
                 bidirectional=opt.brnn)

        self.fertility = opt.fertility
        self.predict_fertility = opt.predict_fertility
        self.supervised_fertility = False
	if 'supervised_fertility' in opt:
            if opt.supervised_fertility:
	        self.supervised_fertility = opt.supervised_fertility

        self.use_sigmoid_fertility = True # False
        if self.predict_fertility:
          if self.use_sigmoid_fertility:
              self.fertility_out = nn.Linear(self.hidden_size * self.num_directions + input_size, 1)
          else:
              self.fertility_linear = nn.Linear(self.hidden_size * self.num_directions + input_size, 2 * self.hidden_size * self.num_directions)
              self.fertility_linear_2 = nn.Linear(2 * self.hidden_size * self.num_directions, 2 * self.hidden_size * self.num_directions)
              self.fertility_out = nn.Linear(2 * self.hidden_size * self.num_directions, 1, bias=False)
        elif self.supervised_fertility:
	  self.sup_linear = nn.Linear(self.hidden_size * self.num_directions + input_size, self.hidden_size)
	  self.sup_linear_2 = nn.Linear(self.hidden_size, 1, bias=False)
	
        self.guided_fertility = opt.guided_fertility

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat
            lengths (LongTensor): batch
            hidden: Initial hidden state.

        Returns:
            hidden_t (FloatTensor): Pair of layers x batch x rnn_size - final
                                    Encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        # CHECKS
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            _, n_batch_ = lengths.size()
            aeq(n_batch, n_batch_)
        # END CHECKS

        emb = self.embeddings(input)
        s_len, n_batch, vec_size = emb.size()

        if self.encoder_layer == "mean":
            # No RNN, just take mean as final state.
            mean = emb.mean(0) \
                   .expand(self.layers, n_batch, vec_size)
            return (mean, mean), emb

        elif self.encoder_layer == "transformer":
            # Self-attention tranformer.
            out = emb.transpose(0, 1).contiguous()
            for i in range(self.layers):
                out = self.transformer[i](out, input[:, :, 0].transpose(0, 1))
            return Variable(emb.data), out.transpose(0, 1).contiguous()

        else:
            #import pdb; pdb.set_trace()
            # Standard RNN encoder.
            packed_emb = emb
            if lengths is not None:
                # Lengths data is wrapped inside a Variable.
                lengths = lengths.data.view(-1).tolist()
                packed_emb = pack(emb, lengths)
            outputs, hidden_t = self.rnn(packed_emb, hidden)
            if lengths:
                outputs = unpack(outputs)[0]
            if self.predict_fertility:
              if self.use_sigmoid_fertility:
                fertility_vals = self.fertility * F.sigmoid(self.fertility_out(torch.cat([outputs.view(-1, self.hidden_size * self.num_directions), emb.view(-1, vec_size)], dim=1)))
              else:
                fertility_vals = F.relu(self.fertility_linear(torch.cat([outputs.view(-1, self.hidden_size * self.num_directions), emb.view(-1, vec_size)], dim=1)))
                fertility_vals = F.relu(self.fertility_linear_2(fertility_vals))
                fertility_vals = 1 + torch.exp(self.fertility_out(fertility_vals))
              fertility_vals = fertility_vals.view(n_batch, s_len)
              #fertility_vals = fertility_vals / torch.sum(fertility_vals, 1).repeat(1, s_len) * s_len
            elif self.guided_fertility:
              fertility_vals = None #evaluation.get_fertility()
	    elif self.supervised_fertility:
              if self.use_sigmoid_fertility:
                #fertility_vals = F.tanh(self.sup_linear(outputs.view(-1, self.hidden_size * self.num_directions)))
                fertility_vals = F.tanh(self.sup_linear(torch.cat([outputs.view(-1, self.hidden_size * self.num_directions), emb.view(-1, vec_size)], dim=1)))
	        fertility_vals = self.sup_linear_2(fertility_vals)
                fertility_vals = self.fertility * F.sigmoid(fertility_vals)
                #fertility_vals = torch.exp(fertility_vals)
                #print fertility_vals
              else:
	        fertility_vals = F.relu(self.sup_linear(outputs.view(-1, self.hidden_size * self.num_directions)))
	        fertility_vals = F.relu(self.sup_linear_2(fertility_vals))
	        fertility_vals = 1 + torch.exp(fertility_vals)
	      fertility_vals = fertility_vals.view(n_batch, s_len)
            else:
              fertility_vals = None
            return hidden_t, outputs, fertility_vals


class Decoder(nn.Module):
    """
    Decoder + Attention recurrent neural network.
    """

    def __init__(self, opt, dicts):
        """
        Args:
            opt: model options
            dicts: Target `Dict` object
        """
        self.layers = opt.layers
        self.decoder_layer = opt.decoder_layer
        self._coverage = opt.coverage_attn
        self.exhaustion_loss = opt.exhaustion_loss
        self.supervised_fertility = False
     	if 'supervised_fertility' in opt:
            if opt.supervised_fertility:
                self.supervised_fertility=opt.supervised_fertility
        self.hidden_size = opt.rnn_size
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts, None)

        if self.decoder_layer == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerDecoder(self.hidden_size, opt)
                 for _ in range(opt.layers)])
        else:
            if opt.rnn_type == "LSTM":
                stackedCell = onmt.modules.StackedLSTM
            else:
                stackedCell = onmt.modules.StackedGRU
            self.rnn = stackedCell(opt.layers, input_size,
                                   opt.rnn_size, opt.dropout)
            self.context_gate = None
            if opt.context_gate is not None:
                self.context_gate = ContextGateFactory(
                    opt.context_gate, opt.word_vec_size,
                    input_size, opt.rnn_size, opt.rnn_size
                )

        self.dropout = nn.Dropout(opt.dropout)
        # Std attention layer.
	if 'c_attn' not in opt:
	    c_attn = 0.0
	else:
	    c_attn = opt.c_attn
        self.attn = onmt.modules.GlobalAttention(
            opt.rnn_size,
            coverage=self._coverage,
            attn_type=opt.attention_type,
            attn_transform=opt.attn_transform,
            c_attn=c_attn
        )
        self.fertility = opt.fertility
        self.predict_fertility = opt.predict_fertility
        self.guided_fertility = opt.guided_fertility
        # Separate Copy Attention.
        self._copy = False
        if opt.copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                opt.rnn_size, attn_type=opt.attention_type)
            self._copy = True

    def compute_max_word_coverage(self, src,
                                  fertility_vals=None,
                                  fert_dict=None,
                                  fert_sents=None,
                                  test=False):
        """
        Forward through the decoder.

        Args:
            input (LongTensor):  (len x batch) -- Input tokens
            src (LongTensor)
            context:  (src_len x batch x rnn_size)  -- Memory bank
            state: an object initializing the decoder.

        Returns:
            outputs: (len x batch x rnn_size)
            final_states: an object of the same form as above
            attns: Dictionary of (src_len x batch)
        """
        s_len, n_batch, _ = src.size()
        if self.predict_fertility:
            max_word_coverage = fertility_vals.clone()
        elif self.guided_fertility:
            fertility_vals = Variable(evaluation.getBatchFertilities(fert_dict, src).transpose(1, 0).contiguous())
            max_word_coverage = fertility_vals
        elif self.supervised_fertility:
            if test:
                max_word_coverage = fertility_vals.clone()
            else:
                fert_tensor_list = [torch.FloatTensor(elem) for elem in fert_sents]
                fert_tensor_list = [evaluation.pad(elem, fertility_vals[i]) for i, elem in enumerate(fert_tensor_list)]
                true_fertility_vals = Variable(torch.stack(fert_tensor_list).cuda(), requires_grad=False)
                max_word_coverage = true_fertility_vals.clone()
        else:
            max_word_coverage = Variable(torch.Tensor([self.fertility]).repeat(n_batch, s_len)).cuda()
        return max_word_coverage


    def forward(self, input, src, context, state,
                max_word_coverage=None,
                test=False):
        """
        Forward through the decoder.

        Args:
            input (LongTensor):  (len x batch) -- Input tokens
            src (LongTensor)
            context:  (src_len x batch x rnn_size)  -- Memory bank
            state: an object initializing the decoder.

        Returns:
            outputs: (len x batch x rnn_size)
            final_states: an object of the same form as above
            attns: Dictionary of (src_len x batch)
        """
        # CHECKS
        t_len, n_batch = input.size()
        s_len, n_batch_, _ = src.size()
        s_len_, n_batch__, _ = context.size()
        aeq(n_batch, n_batch_, n_batch__)

        # aeq(s_len, s_len_)
        # END CHECKS
        if self.decoder_layer == "transformer":
            if state.previous_input:
                input = torch.cat([state.previous_input.squeeze(2), input], 0)
        emb = self.embeddings(input.unsqueeze(2))
        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []

        # Setup the different types of attention.
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []
        if self.exhaustion_loss:
            attns["upper_bounds"] = []
        if self.decoder_layer == "transformer":
            # Tranformer Decoder.
            assert isinstance(state, TransformerDecoderState)
            output = emb.transpose(0, 1).contiguous()
            src_context = context.transpose(0, 1).contiguous()
            for i in range(self.layers):
                output, attn \
                    = self.transformer[i](output, src_context,
                                          src[:, :, 0].transpose(0, 1),
                                          input.transpose(0, 1))
            outputs = output.transpose(0, 1).contiguous()
            if state.previous_input:
                outputs = outputs[state.previous_input.size(0):]
                attn = attn[:, state.previous_input.size(0):].squeeze()
                attn = torch.stack([attn])
            attns["std"] = attn
            if self._copy:
                attns["copy"] = attn
            state = TransformerDecoderState(input.unsqueeze(2))
        else:
            assert isinstance(state, RNNDecoderState)
            output = state.input_feed.squeeze(0)
            hidden = state.hidden
            cumulative_attention = state.cumulative_attention
            # CHECKS
            n_batch_, _ = output.size()
            aeq(n_batch, n_batch_)
            # END CHECKS

            coverage = state.coverage.squeeze(0) \
                if state.coverage is not None else None

            # Standard RNN decoder.
            for i, emb_t in enumerate(emb.split(1)):
                # Initialize cumulative attention for the current batch.
                if cumulative_attention is None:
                    cumulative_attention = Variable(
                        torch.Tensor([0]).repeat(n_batch_, s_len_).cuda())

                emb_t = emb_t.squeeze(0)
                if self.input_feed:
                    emb_t = torch.cat([emb_t, output], 1)

                if max_word_coverage is not None:
                    upper_bounds = max_word_coverage - cumulative_attention
                    # Use <SINK> token for absorbing remaining attention weight.
                    upper_bounds[:, -1] = Variable(
                        torch.ones(upper_bounds.size(0)))
                    # Make sure we're >= 0.
                    upper_bounds = torch.max(upper_bounds,
                                             Variable(torch.zeros(
                                                 upper_bounds.size(0),
                                                 upper_bounds.size(1)).cuda()))
                else:
                    upper_bounds = None

                rnn_output, hidden = self.rnn(emb_t, hidden)
                attn_output, attn = self.attn(rnn_output,
                                              context.transpose(0, 1),
                                              upper_bounds=upper_bounds)

                cumulative_attention += attn

                if self.context_gate is not None:
                    output = self.context_gate(
                        emb_t, rnn_output, attn_output
                    )
                    output = self.dropout(output)
                else:
                    output = self.dropout(attn_output)
                outputs += [output]
                attns["std"] += [attn]

                # COVERAGE
                if self._coverage:
                    coverage = (coverage + attn) if coverage else attn
                    attns["coverage"] += [coverage]

                # COPY
                if self._copy:
                    _, copy_attn = self.copy_attn(output,
                                                  context.transpose(0, 1))
                    attns["copy"] += [copy_attn]
                if self.exhaustion_loss:
                    attns["upper_bounds"] += [upper_bounds]
            #if self.supervised_fertility:
            #    if not test:
            #        attns["true_fertility_vals"] += [max_word_coverage]
            #    attns["predicted_fertility_vals"] += [fertility_vals]
            state = RNNDecoderState(hidden, output.unsqueeze(0),
                                    coverage.unsqueeze(0)
                                    if coverage is not None else None,
                                    cumulative_attention)
            outputs = torch.stack(outputs)
            for k in attns:
                attns[k] = torch.stack(attns[k])
        return outputs, state, attns


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim
        We need to convert it to layers x batch x (directions*dim)
        """
        if self.encoder.num_directions == 2:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        if self.decoder.decoder_layer == "transformer":
            return TransformerDecoderState()
        elif isinstance(enc_hidden, tuple):
            dec = RNNDecoderState(tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:
            dec = RNNDecoderState(self._fix_enc_hidden(enc_hidden))
        dec.init_input_feed(context, self.decoder.hidden_size)
        return dec

    def forward(self, src, tgt, lengths, dec_state=None, fert_dict=None, fert_sents=None, test=False):
        """
        Args:
            src, tgt, lengths
            dec_state: A decoder state object

        Returns:
            outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        #print("src:", src)
        enc_hidden, context, fertility_vals = self.encoder(src, lengths)
        enc_state = self.init_decoder_state(context, enc_hidden)
        max_word_coverage = self.decoder.compute_max_word_coverage(
            src, fertility_vals, fert_dict, fert_sents, test=test)
        out, dec_state, attns = self.decoder(tgt, src, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             max_word_coverage,
                                             #fertility_vals,
                                             test=test)
        if fertility_vals is not None:
            attns["predicted_fertility_vals"] = torch.stack([fertility_vals])
        attns["true_fertility_vals"] = torch.stack([max_word_coverage])
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    def detach(self):
        for h in self.all:
            if h is not None:
                h.detach_()

    def repeatBeam_(self, beamSize):
        self._resetAll([Variable(e.data.repeat(1, beamSize, 1))
                        for e in self.all])

    def beamUpdate_(self, idx, positions, beamSize):
        for e in self.all:
            a, br, d = e.size()
            sentStates = e.view(a, beamSize, br // beamSize, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))

class RNNDecoderState(DecoderState):
    def __init__(self, rnnstate, input_feed=None, coverage=None,
                 cumulative_attention=None):
        # all objects are X x batch x dim
        # or X x (beam * sent) for beam search
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage
        self.cumulative_attention = cumulative_attention
        self.all = self.hidden + (self.input_feed,)

    def init_input_feed(self, context, rnn_size):
        batch_size = context.size(1)
        h_size = (batch_size, rnn_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)
        self.all = self.hidden + (self.input_feed,)

    def _resetAll(self, all):
        vars = [Variable(a.data if isinstance(a, Variable) else a,
                         volatile=True) for a in all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
        self.all = self.hidden + (self.input_feed,)

    def beamUpdate_(self, idx, positions, beamSize):
        # I'm overriding this method to handle the upper bounds in the beam
        # updates. May be simpler to add this as part of self.all and not
        # do the overriding.
        #import pdb; pdb.set_trace()
        DecoderState.beamUpdate_(self, idx, positions, beamSize)
        if self.cumulative_attention is not None:
            e = self.cumulative_attention
            br, d = e.size()
            sentStates = e.view(beamSize, br // beamSize, d)[:, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(0, positions))

class TransformerDecoderState(DecoderState):
    def __init__(self, input=None):
        # all objects are X x batch x dim
        # or X x (beam * sent) for beam search
        self.previous_input = input
        self.all = (self.previous_input,)

    def _resetAll(self, all):
        vars = [(Variable(a.data if isinstance(a, Variable) else a,
                          volatile=True))
                for a in all]
        self.previous_input = vars[0]
        self.all = (self.previous_input,)

    def repeatBeam_(self, beamSize):
        pass
