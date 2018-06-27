"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder, LanguageModel, \
                        CharEmbeddingsCNN, APEModel, APEInputFeedRNNDecoder
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, TransformerDecoder, \
                         CNNEncoder, CNNDecoder, AudioEncoder, ELMo
from onmt.Utils import use_gpu
from torch.nn.init import xavier_uniform


def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True,
                    elmo=None):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
        elmo(nn.Module): ELMo embedding extension.
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    elif opt.lm:
        if opt.lm_use_char_input:
            embedding_dim = opt.lm_char_vec_size
        else:
            embedding_dim = opt.lm_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      sparse=opt.optim == "sparseadam",
                      elmo=elmo)


def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                          opt.rnn_size, opt.dropout, embeddings,
                          opt.bridge)


def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.model_type == "ape":
        if opt.decoder_type in ["transformer", "cnn"]:
            raise NotImplementedError("Only rnn is implemented for APE task")
        return APEInputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                      opt.dec_layers, opt.rnn_size,
                                      opt.global_attention,
                                      opt.coverage_attn,
                                      opt.context_gate,
                                      opt.copy_attn,
                                      opt.dropout,
                                      embeddings,
                                      opt.reuse_copy_attn)
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn)
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings,
                             opt.reuse_copy_attn)


def load_test_model(opt, dummy_opt):
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']

    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type,
        use_char=model_opt.elmo)

    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields,
                            use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio", "ape"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    if model_opt.elmo:
        print('Building language model for ELMo...')
        lm_checkpoint = torch.load(model_opt.bilm_src_path,
                                   map_location=lambda storage, loc: storage)
        lm_opt = lm_checkpoint['opt']

        # Implementation errors
        if not lm_opt.lm_use_char_input:
            raise NotImplementedError("The Language Model used in ELMo needs"
                                      "to have character-based input")
        if not lm_opt.bilm:
            raise NotImplementedError("The Language Model used in ELMo needs"
                                      "to be bidirectional")
        if model_opt.encoder_type != 'rnn':
            raise NotImplementedError("ELMo is only implemented for RNN "
                                      "Encoder.")

        # Load the language model without the generator
        language_model = make_language_model(lm_opt, fields, gpu,
                                             lm_checkpoint,
                                             'src',
                                             use_generator=False)

        elmo_src = ELMo(language_model, model_opt.elmo_dropout)

        if model_opt.model_type == 'ape' and model_opt.bilm_mt_path:
            lm_checkpoint = torch.load(model_opt.bilm_mt_path,
                                       map_location=lambda storage,
                                       loc: storage)
            lm_opt = lm_checkpoint['opt']

            # Implementation errors
            if not lm_opt.lm_use_char_input:
                raise NotImplementedError(
                    "The Language Model used in ELMo needs "
                    "to have character-based input")
            if not lm_opt.bilm:
                raise NotImplementedError(
                    "The Language Model used in ELMo needs "
                    "to be bidirectional")
            if model_opt.encoder_type != 'rnn':
                raise NotImplementedError("ELMo is only implemented for RNN "
                                          "Encoder.")

            language_model = make_language_model(lm_opt, fields, gpu,
                                                 lm_checkpoint,
                                                 'src',
                                                 use_generator=False)

            elmo_mt = ELMo(language_model, model_opt.elmo_dropout)
        else:
            elmo_mt = None
    else:
        elmo_src = None
        elmo_mt = None

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts, elmo=elmo_src)
        encoder = make_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)
    elif model_opt.model_type == 'ape':
        src_dict = fields["src"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts, elmo=elmo_src)
        encoder_src = make_encoder(model_opt, src_embeddings)
        mt_dict = fields["mt"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'mt')
        mt_embeddings = make_embeddings(model_opt, mt_dict,
                                        feature_dicts, elmo=elmo_mt)
        encoder_mt = make_encoder(model_opt, mt_embeddings)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    # Share the embedding matrix of mt and tgt for ape models
    if model_opt.model_type == 'ape':
        tgt_embeddings.word_lut.weight = mt_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    if model_opt.model_type == 'ape':
        model = APEModel(encoder_src, encoder_mt, decoder)
    else:
        model = NMTModel(encoder, decoder)
    model.model_type = model_opt.model_type

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax(dim=-1))
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)

        if model_opt.model_type == 'text':
            if hasattr(model.encoder, 'embeddings'):
                model.encoder.embeddings.load_pretrained_vectors(
                        model_opt.pre_word_vecs_enc,
                        model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def make_language_model(model_opt, fields, gpu, checkpoint=None,
                        side='tgt', use_generator=True):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the Language Model.
    """
    assert model_opt.model_type in ["text"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make Embeddings
    word_dict = fields[side].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, side)
    # Get Padding idx in case bidirectional language model is necessary
    padding_idx = word_dict.stoi[onmt.io.PAD_WORD]

    if model_opt.lm_use_char_input:
        if "char_"+side not in fields.keys():
            raise ValueError("There is no character vocabulary field"
                             " available. Please preprocess data with"
                             " -use_char flag.")
        # Load character vocabulary
        char_word_dict = fields["char_"+side].nesting_field.vocab
        # Create character embeddings
        char_embeddings = make_embeddings(model_opt, char_word_dict,
                                          feature_dicts, for_encoder=False)
        # Initialize Convolutions, Highway Layer and Projection
        # into word embedding size
        embeddings = CharEmbeddingsCNN(model_opt, char_embeddings)

    else:
        embeddings = make_embeddings(model_opt, word_dict,
                                     feature_dicts, for_encoder=False)

    # Make LanguageModel.
    model = LanguageModel(model_opt, embeddings, gpu, padding_idx)
    model.model_type = model_opt.model_type

    # Save model options as a boolean in the model
    if model_opt.bilm:
        model.bidirectional = True
    else:
        model.bidirectional = False

    if model_opt.lm_use_char_input:
        model.char_convs = True
    else:
        model.char_convs = False

    # Make Generator.
    output_size = model_opt.lm_word_vec_size if model_opt.lm_use_projection \
        else model_opt.lm_rnn_size

    if use_generator:
        generator = nn.Sequential(
            nn.Linear(output_size, len(fields[side].vocab)),
            nn.LogSoftmax(dim=-1))

    if model_opt.tie_weights:
            if model_opt.lm_use_char_input:
                raise ValueError('It is not possible to use this flag '
                                 'when using character input embeddings.')
            if output_size != model_opt.lm_word_vec_size:
                raise ValueError(
                    'When using the tied flag, hidden size'
                    ' must be equal to embedding size.')
            generator[0].weight = embeddings.word_lut.weight

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        if use_generator:
            generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            if use_generator:
                for p in generator.parameters():
                    p.data.uniform_(-model_opt.param_init,
                                    model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform(p)
            if use_generator:
                for p in generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform(p)

        # if hasattr(model, 'embeddings'):
        #     model.load_pretrained_vectors(
        #             model_opt.pre_word_vecs, model_opt.fix_word_vecs)

    # Add generator to model (this registers it as parameter of model).
    if use_generator:
        model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
