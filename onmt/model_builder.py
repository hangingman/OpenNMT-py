"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder

from onmt.decoders.decoder import InputFeedRNNDecoder, StdRNNDecoder,\
                                  APEInputFeedRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder

from onmt.modules import Embeddings, CopyGenerator, CharEmbeddingsCNN,\
                         SampledSoftmax, ELMo, ExtendedEmbeddings
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger


def build_embeddings(opt, word_dict, feature_dicts, for_encoder=True,
                     elmo=None):
    """
    Build an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): build Embeddings for encoder or decoder?
        elmo(nn.Module): ELMo embedding extension.
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    elif opt.lm:
        if opt.use_char_input:
            embedding_dim = opt.lm_char_vec_size
        else:
            embedding_dim = opt.lm_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[inputters.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[inputters.PAD_WORD]
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


def build_elmo(opt, fields, gpu, side='src',
               forward_only=False, reused_bilm=None):
    if side == 'src':
        bilm_path = opt.src_bilm_path
    elif side == 'mt':
        bilm_path = opt.mt_bilm_path

    if reused_bilm is None:
        lm_checkpoint = torch.load(bilm_path,
                                   map_location=lambda storage, loc: storage)
        lm_opt = lm_checkpoint['opt']
        if not lm_opt.use_char_input:
            raise NotImplementedError("The Language Model used in ELMo needs "
                                      "to have character-based input")
        if not lm_opt.lm_use_bidir:
            raise NotImplementedError("The Language Model used in ELMo needs "
                                      "to be bidirectional")
        language_model = build_language_model(lm_opt, fields, gpu,
                                              lm_checkpoint,
                                              side,
                                              use_generator=False)
    else:
        language_model = reused_bilm

    elmo = ELMo(language_model, opt.elmo_dropout, forward_only)

    return elmo


def build_ext_embeddings(opt, word_dict, feature_dicts, base_embds,
                         for_encoder=True):
    """
    build an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[inputters.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[inputters.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return ExtendedEmbeddings(base_embds, word_vec_size=embedding_dim,
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
                              elmo=base_embds.elmo)


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.heads, opt.transformer_ff,
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


def build_decoder(opt, embeddings, elmo=None):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.ape:
        return APEInputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                      opt.dec_layers, opt.rnn_size,
                                      opt.global_attention,
                                      opt.global_attention_function,
                                      opt.coverage_attn,
                                      opt.context_gate,
                                      opt.copy_attn,
                                      opt.dropout,
                                      embeddings,
                                      opt.reuse_copy_attn,
                                      elmo=elmo)
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.heads, opt.transformer_ff,
                                  opt.global_attention, opt.copy_attn,
                                  opt.self_attn_type,
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
                                   opt.global_attention_function,
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
                             opt.global_attention_function,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings,
                             opt.reuse_copy_attn)


def build_affine_proj(tgt_dict, linear_layer):
    affine_proj = linear_layer
    # FIXME initialization!
    new_proj = nn.Linear(affine_proj.in_features, len(tgt_dict))
    new_proj.weight.data[0:affine_proj.out_features] = affine_proj.weight.data
    new_proj.bias.data[:affine_proj.out_features] = affine_proj.bias.data
    return new_proj


def extend_vocabulary(model, fields, ext_fields, model_opt):
    # Source side extension
    # Extend src vocabulary
    if not model_opt.lm:
        fields["src"].vocab.extend(ext_fields["src"].vocab)
        # Collect vocab and features
        src_dict = fields["src"].vocab
        feature_dicts = inputters.collect_feature_vocabs(fields, 'src')
        # Retrieve the previous embeddings layer and create an extension
        src_embs = model.encoder.embeddings
        src_embs_ext = build_ext_embeddings(model_opt, src_dict, feature_dicts,
                                            src_embs)
        # Set the new embeddings layer
        model.encoder.embeddings = src_embs_ext
    # Target side extension
    # Extend tgt vocabulary
    fields["tgt"].vocab.extend(ext_fields["tgt"].vocab)
    logger.info('Vocabulary extended to ' + str(len(fields["tgt"].vocab)))
    # Collect vocab and features
    tgt_dict = fields["tgt"].vocab
    feature_dicts = inputters.collect_feature_vocabs(fields, 'tgt')
    if hasattr(model, "decoder"):
        # Retrieve the previous embeddings layer and create an extension
        tgt_embs = model.decoder.embeddings
        tgt_embs_ext = build_ext_embeddings(model_opt, tgt_dict, feature_dicts,
                                            tgt_embs)
        # Set the new embeddings layer
        model.decoder.embeddings = tgt_embs_ext
    # Set the generator softmax size
    if not model_opt.lm_use_sampled_softmax:
        affine_proj = build_affine_proj(tgt_dict, model.generator[0])
        model.generator = nn.Sequential(affine_proj,
                                        nn.LogSoftmax())
    else:
        output_size = model_opt.lm_word_vec_size if \
                            model_opt.lm_use_projection \
                            else model_opt.lm_rnn_size
        model.generator = SampledSoftmax(len(fields["tgt"].vocab),
                                         model_opt.lm_n_samples_softmax,
                                         output_size)

        affine_proj = build_affine_proj(tgt_dict, model.generator.params)
        model.generator.params = affine_proj


def load_test_model(opt, dummy_opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    fields = inputters.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type,
        use_char=checkpoint['opt'].use_char_input)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    if model_opt.lm:
        model = build_language_model(model_opt, fields, use_gpu(opt),
                                     checkpoint)
    else:
        model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None, ext_fields=None):
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
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Build encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = inputters.collect_feature_vocabs(fields, 'src')

        elmo = build_elmo(model_opt, fields,
                          gpu, 'src') if model_opt.use_elmo else None

        src_embeddings = build_embeddings(model_opt, src_dict, feature_dicts,
                                          elmo=elmo)
        encoder = build_encoder(model_opt, src_embeddings)

        if model_opt.ape:
            encoder_src = encoder
            del encoder

            mt_dict = fields["mt"].vocab
            feature_dicts = inputters.collect_feature_vocabs(fields, 'mt')
            elmo = build_elmo(model_opt, fields,
                              gpu, 'mt') if model_opt.use_elmo else None
            mt_embeddings = build_embeddings(model_opt, mt_dict,
                                             feature_dicts,
                                             elmo=elmo)
            encoder_mt = build_encoder(model_opt, mt_embeddings)

    elif model_opt.model_type == "img":
        if ("image_channel_size" not in model_opt.__dict__):
            image_channel_size = 3
        else:
            image_channel_size = model_opt.image_channel_size

        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               image_channel_size)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Build decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = inputters.collect_feature_vocabs(fields, 'tgt')

    dec_elmo = build_elmo(
        model_opt, fields,
        gpu, 'tgt',
        forward_only=True,
        reused_bilm=elmo.lang_model) if model_opt.use_decoder_elmo else None

    tgt_embeddings = build_embeddings(model_opt, tgt_dict,
                                      feature_dicts, for_encoder=False,
                                      elmo=dec_elmo)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    # Share the embedding matrix of mt and tgt for ape models
    if model_opt.ape and not model_opt.pretrained_softmax_path:
        tgt_embeddings.word_lut.weight = mt_embeddings.word_lut.weight

    dec_out_elmo = build_elmo(
        model_opt, fields,
        gpu, 'tgt',
        forward_only=True,
        reused_bilm=dec_elmo.lang_model)\
        if model_opt.use_dec_out_elmo else None

    decoder = build_decoder(model_opt, tgt_embeddings, dec_out_elmo)

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda" if gpu else "cpu")
    if model_opt.ape:
        model = onmt.models.APEModel(encoder_src, encoder_mt, decoder)
    else:
        model = onmt.models.NMTModel(encoder, decoder)
    model.model_type = model_opt.model_type

    output_size = model_opt.rnn_size
    if model_opt.use_dec_out_elmo:
        output_size += dec_out_elmo.lang_model.input_size

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(output_size, len(fields["tgt"].vocab)), gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(output_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if model_opt.pretrained_softmax_path:
            lm_checkpoint = torch.load(
                                model_opt.pretrained_softmax_path,
                                map_location=lambda storage, loc: storage)
            lm_generator = lm_checkpoint['generator']

            # UNK symbol
            generator.state_dict()['0.weight'].data[0:1].copy_(
                lm_generator['params.weight'][0:1])
            generator.state_dict()['0.bias'].data[0:1].copy_(
                lm_generator['params.bias'][0:1])
            # Skip padding (LM doesn't have padding) and
            # replace the remaining rows
            generator.state_dict()['0.weight'].data[
                2:lm_generator['params.weight'].shape[0]+1].copy_(
                    lm_generator['params.weight'][1:])
            generator.state_dict()['0.bias'].data[
                2:lm_generator['params.bias'].shape[0]+1].copy_(
                    lm_generator['params.bias'][1:])

        if not model_opt.ape and hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator
    if ext_fields is not None:
        extend_vocabulary(model, fields, ext_fields, model_opt)
    model.to(device)

    return model


def build_language_model(model_opt, fields, gpu,
                         checkpoint=None, side='tgt', use_generator=True,
                         ext_fields=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        side: which input field should we use. Default is tgt since
            since that's what we use during training.
    Returns:
        the Language Model.
    """
    assert model_opt.model_type in ["text"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make Embeddings
    word_dict = fields[side].vocab
    feature_dicts = inputters.collect_feature_vocabs(fields, side)

    if model_opt.use_char_input:
        if "char_"+side not in fields.keys():
            raise ValueError("There is no character vocabulary field"
                             " available. Please preprocess data with"
                             " -use_char flag.")
        # Load character vocabulary
        char_word_dict = fields["char_"+side].nesting_field.vocab
        # Create character embeddings
        char_embeddings = build_embeddings(model_opt, char_word_dict,
                                           feature_dicts, for_encoder=False)
        # Initialize Convolutions, Highway Layer and Projection
        # into word embedding size
        embeddings = CharEmbeddingsCNN(model_opt, char_embeddings)

    else:
        embeddings = build_embeddings(model_opt, word_dict,
                                      feature_dicts, for_encoder=False)

    # Make LanguageModel.
    device = torch.device("cuda" if gpu else "cpu")
    model = onmt.models.LanguageModel(model_opt, embeddings)
    model.model_type = model_opt.model_type

    # Save model options as a boolean in the model
    if model_opt.lm_use_bidir:
        model.bidirectional = True
    else:
        model.bidirectional = False

    if model_opt.use_char_input:
        model.char_convs = True
    else:
        model.char_convs = False

    # Make Generator.
    output_size = model_opt.lm_word_vec_size if model_opt.lm_use_projection \
        else model_opt.lm_rnn_size

    if use_generator:
        if not model_opt.lm_use_sampled_softmax:
            generator = nn.Sequential(
                nn.Linear(output_size, len(fields[side].vocab)),
                nn.LogSoftmax(dim=-1))
        else:
            generator = SampledSoftmax(len(fields[side].vocab),
                                       model_opt.lm_n_samples_softmax,
                                       output_size)

    if model_opt.lm_tie_weights:
        if model_opt.use_char_input:
            raise ValueError('It is not possible to tie weights '
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
                    xavier_uniform_(p)
            if use_generator:
                for p in generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

    # Add generator to model (this registers it as parameter of model).
    if use_generator:
        model.generator = generator
        if ext_fields is not None:
            extend_vocabulary(model, fields, ext_fields, model_opt)

    model.to(device)

    return model


def build_model(model_opt, opt, fields, ext_fields=None, checkpoint=None):
    """ Build the Model """
    logger.info('Building model...')
    if model_opt.lm:
        model = build_language_model(model_opt, fields,
                                     use_gpu(opt), checkpoint,
                                     ext_fields=ext_fields)
    else:
        model = build_base_model(model_opt, fields,
                                 use_gpu(opt), checkpoint,
                                 ext_fields=ext_fields)
    logger.info(model)
    return model
