# -*- coding: utf-8 -*-
"""
    Defining general functions for inputters
"""
import glob
import os
from collections import Counter, defaultdict, OrderedDict
from itertools import count
import sys

import torch
import torchtext.data
import torchtext.vocab

from onmt.inputters.dataset_base import UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD
from onmt.inputters.text_dataset import TextDataset, MonotextDataset, \
                                        APETextDataset
from onmt.inputters.image_dataset import ImageDataset
from onmt.inputters.audio_dataset import AudioDataset
from onmt.utils.logging import logger

import gc


def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


torchtext.vocab.Vocab.__getstate__ = _getstate
torchtext.vocab.Vocab.__setstate__ = _setstate


def get_fields(data_type, n_src_features, n_tgt_features, use_char=False,
               n_mt_features=None):
    """
    Args:
        data_type: type of the source input. Options are [text|img|audio].
        n_src_features: the number of source features to
            create `torchtext.data.Field` for.
        n_tgt_features: the number of target features to
            create `torchtext.data.Field` for.
        use_char: boolean to decide if character fields are necessary

    Returns:
        A dictionary whose keys are strings and whose values are the
        corresponding Field objects.
    """
    if n_mt_features is not None:
        # APE task
        return APETextDataset.get_fields(n_src_features, n_mt_features,
                                         n_tgt_features, use_char)
    elif data_type == 'text':
        return TextDataset.get_fields(n_src_features, n_tgt_features, use_char)
    elif data_type == 'monotext':
        return MonotextDataset.get_fields(n_tgt_features, use_char)
    elif data_type == 'img':
        return ImageDataset.get_fields(n_src_features, n_tgt_features)
    elif data_type == 'audio':
        return AudioDataset.get_fields(n_src_features, n_tgt_features)
    else:
        raise ValueError("Data type not implemented")


def load_fields_from_vocab(vocab, data_type="text", use_char=False):
    """
    Load Field objects from `vocab.pt` file.
    """
    vocab = dict(vocab)
    if not any(['_nested' in k for k in vocab.keys()]) and use_char:
        raise ValueError('There are no character fields '
                         'in the vocab file. Use -use_char '
                         'during preprocessing to be able to use characters.')
    if data_type == "monotext":
        n_src_features = 0
    else:
        n_src_features = len(collect_features(vocab, 'src'))

    # In case we are doing the APE task
    if 'mt' in vocab.keys():
        n_mt_features = len(collect_features(vocab, 'mt'))
    else:
        n_mt_features = None

    n_tgt_features = len(collect_features(vocab, 'tgt'))
    fields = get_fields(data_type, n_src_features, n_tgt_features,
                        use_char, n_mt_features)
    for k, v in vocab.items():
        if "_nested" in k:
            # build the vocabulary for the nested character field.
            # when using character fields, there are two vocabs:
            # `vocab` and `vocab_nested`. But in the fields object,
            # the nested character field is an attribute of the main field
            # instead of a separate field. So, if we find a nested
            # vocabulary, we need to associate it with the correct
            # nesting_field.
            k = k[:-7]
            v.stoi = defaultdict(lambda: 0, v.stoi)
            fields[k].nesting_field.vocab = v

        else:
            # Hack. Can't pickle defaultdict :(
            v.stoi = defaultdict(lambda: 0, v.stoi)
            try:
                fields[k].vocab = v
            except KeyError:
                if 'char' in k:
                    raise ValueError('If using character input,'
                                     ' please use'
                                     ' -use_char_input option.')
                sys.exit()
    return fields


def save_fields_to_vocab(fields):
    """
    Save Vocab objects in Field objects to `vocab.pt` file.
    """
    vocab = []
    for k, f in fields.items():
        if f is not None and 'vocab' in f.__dict__:
            f.vocab.stoi = f.vocab.stoi
            vocab.append((k, f.vocab))
            # Check if the field we are taking a vocab from also has
            # a nesting field, and in that case, save that vocab as well
            if 'nesting_field' in dir(f):
                # If it exists, also save the nested vocab
                vocab.append((k + '_nested', f.nesting_field.vocab))
    return vocab


def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.

    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return torchtext.vocab.Vocab(merged,
                                 specials=[UNK_WORD, PAD_WORD,
                                           BOS_WORD, EOS_WORD],
                                 max_size=vocab_size)


def get_num_features(data_type, corpus_file, side):
    """
    Args:
        data_type (str): type of the source input.
            Options are [text|img|audio].
        corpus_file (str): file path to get the features.
        side (str): for source or for target.

    Returns:
        number of features on `side`.
    """
    assert side in ["src", "tgt", "mt"]

    if data_type == 'text':
        return TextDataset.get_num_features(corpus_file, side)
    elif data_type == 'monotext':
        # no features used in language modelling
        return 0
    elif data_type == 'img':
        return ImageDataset.get_num_features(corpus_file, side)
    elif data_type == 'audio':
        return AudioDataset.get_num_features(corpus_file, side)
    else:
        raise ValueError("Data type not implemented")


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Tensor): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src', 'mt', 'tgt', 'char_src', 'char_mt', 'char_tgt']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    if data_type == 'text':
        return torch.cat([level.unsqueeze(2) for level in levels], 2)
    else:
        return levels[0]


def collect_features(fields, side="src"):
    """
    Collect features from Field object.
    """
    assert side in ["src", "mt", "tgt"]
    feats = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feats.append(key)
    return feats


def collect_feature_vocabs(fields, side):
    """
    Collect feature Vocab objects from Field object.
    """
    assert side in ['src', 'mt', 'tgt']
    feature_vocabs = []
    for j in count():
        key = side + "_feat_" + str(j)
        if key not in fields:
            break
        feature_vocabs.append(fields[key].vocab)
    return feature_vocabs


def build_dataset(fields, data_type, src_data_iter=None, src_path=None,
                  src_dir=None, tgt_data_iter=None, tgt_path=None,
                  src_seq_length=0, tgt_seq_length=0,
                  src_seq_length_trunc=0, tgt_seq_length_trunc=0,
                  dynamic_dict=True, sample_rate=0,
                  window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True,
                  image_channel_size=3,
                  mt_data_iter=None, mt_path=None, mt_seq_length=0,
                  mt_seq_length_trunc=0):
    """
    Build src/tgt examples iterator from corpus files, also extract
    number of features.
    """

    def _make_examples_nfeats_tpl(data_type, src_data_iter, src_path, src_dir,
                                  src_seq_length_trunc, sample_rate,
                                  window_size, window_stride,
                                  window, normalize_audio,
                                  image_channel_size=3):
        """
        Process the corpus into (example_dict iterator, num_feats) tuple
        on source side for different 'data_type'.
        """

        if data_type == 'text':
            src_examples_iter, num_src_feats = \
                TextDataset.make_text_examples_nfeats_tpl(
                    src_data_iter, src_path, src_seq_length_trunc, "src")

        elif data_type == 'img':
            src_examples_iter, num_src_feats = \
                ImageDataset.make_image_examples_nfeats_tpl(
                    src_data_iter, src_path, src_dir, image_channel_size)

        elif data_type == 'audio':
            if src_data_iter:
                raise ValueError("""Data iterator for AudioDataset isn't
                                    implemented""")

            if src_path is None:
                raise ValueError("AudioDataset requires a non None path")
            src_examples_iter, num_src_feats = \
                AudioDataset.make_audio_examples_nfeats_tpl(
                    src_path, src_dir, sample_rate,
                    window_size, window_stride, window,
                    normalize_audio)

        return src_examples_iter, num_src_feats

    if data_type != 'monotext':
        src_examples_iter, num_src_feats = \
            _make_examples_nfeats_tpl(data_type, src_data_iter, src_path,
                                      src_dir, src_seq_length_trunc,
                                      sample_rate, window_size, window_stride,
                                      window, normalize_audio,
                                      image_channel_size=image_channel_size)
    else:
        src_examples_iter = None
        num_src_feats = 0

    if mt_path is not None:
        mt_examples_iter, num_mt_feats = \
            APETextDataset.make_text_examples_nfeats_tpl(
                mt_data_iter, mt_path, mt_seq_length_trunc, "mt")
    else:
        mt_examples_iter = num_mt_feats = None

    # For all data types, the tgt side corpus is in form of text.
    tgt_examples_iter, num_tgt_feats = \
        TextDataset.make_text_examples_nfeats_tpl(
            tgt_data_iter, tgt_path, tgt_seq_length_trunc, "tgt")

    if mt_examples_iter is not None:
        dataset = APETextDataset(fields,
                                 src_examples_iter, mt_examples_iter,
                                 tgt_examples_iter,
                                 num_src_feats, num_mt_feats, num_tgt_feats,
                                 src_seq_length=src_seq_length,
                                 mt_seq_length=mt_seq_length,
                                 tgt_seq_length=tgt_seq_length,
                                 dynamic_dict=dynamic_dict,
                                 use_filter_pred=use_filter_pred)
    elif 'text' in data_type:
        dataset = TextDataset(fields, src_examples_iter, tgt_examples_iter,
                              num_src_feats, num_tgt_feats,
                              src_seq_length=src_seq_length,
                              tgt_seq_length=tgt_seq_length,
                              dynamic_dict=dynamic_dict,
                              use_filter_pred=use_filter_pred)

    elif data_type == 'img':
        dataset = ImageDataset(fields, src_examples_iter, tgt_examples_iter,
                               num_src_feats, num_tgt_feats,
                               tgt_seq_length=tgt_seq_length,
                               use_filter_pred=use_filter_pred,
                               image_channel_size=image_channel_size)

    elif data_type == 'audio':
        dataset = AudioDataset(fields, src_examples_iter, tgt_examples_iter,
                               num_src_feats, num_tgt_feats,
                               tgt_seq_length=tgt_seq_length,
                               sample_rate=sample_rate,
                               window_size=window_size,
                               window_stride=window_stride,
                               window=window,
                               normalize_audio=normalize_audio,
                               use_filter_pred=use_filter_pred)

    return dataset


def _build_field_vocab(field, counter, char_counter=None, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)

    # Check if this field has an associated nested field for characters
    if hasattr(field, 'nesting_field'):
        # The current field has a nested character field, so we build
        # a vocabulary for the nested field as well
        char_specials = list(OrderedDict.fromkeys(
            tok for tok in [field.nesting_field.unk_token,
                            field.nesting_field.pad_token,
                            field.nesting_field.init_token,
                            field.nesting_field.eos_token]
            if tok is not None))
        # Make sure special tokens are not repeated
        for special in specials:
            if special not in char_specials:
                char_specials.append(special)
        # Create the vocab for the nested field
        field.nesting_field.vocab = field.nesting_field.vocab_cls(
                                char_counter,
                                specials=char_specials,
                                **kwargs)


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency,
                mt_vocab_path, mt_vocab_size, mt_words_min_frequency):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_path(string): Path to src vocabulary file.
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """

    if data_type == 'monotext':
        return build_vocab_mono(train_dataset_files, fields,
                                tgt_vocab_path, tgt_vocab_size,
                                tgt_words_min_frequency)
    counter = {}

    # Prop src from field to get lower memory using when training with image
    if data_type == 'img':
        fields.pop("src")

    for k in fields:
        counter[k] = Counter()

    # Load vocabulary
    src_vocab = load_vocabulary(src_vocab_path, tag="source")
    mt_vocab = load_vocabulary(mt_vocab_path, tag="mt")
    tgt_vocab = load_vocabulary(tgt_vocab_path, tag="target")

    if "char_src" in fields.keys():
        n_chars = 256
        for idx in range(n_chars):
            # Create a counter for characters. By doing (n_chars - idx)
            # we are preserving the Unicode order of the characters by
            # giving the frequencies in descending order
            counter["char_src"][chr(idx)] = n_chars - idx

    if "char_mt" in fields.keys():
        n_chars = 256
        for idx in range(n_chars):
            # Create a counter for characters. By doing (n_chars - idx)
            # we are preserving the Unicode order of the characters by
            # giving the frequencies in descending order
            counter["char_mt"][chr(idx)] = n_chars - idx

    if "char_tgt" in fields.keys():
        n_chars = 256
        for idx in range(n_chars):
            # Create a counter for characters. By doing (n_chars - idx)
            # we are preserving the Unicode order of the characters by
            # giving the frequencies in descending order
            counter["char_tgt"][chr(idx)] = n_chars - idx

    for index, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for k in fields:
                val = getattr(ex, k, None)
                if val is not None and not fields[k].sequential:
                    val = [val]
                elif val is not None and 'char' in k:
                    # ignore character fields, these were already
                    # initialized to be all unicode characters from 0 to 255.
                    continue
                elif k == 'src' and src_vocab:
                    val = [item for item in val if item in src_vocab]
                elif k == 'tgt' and tgt_vocab:
                    val = [item for item in val if item in tgt_vocab]
                elif k == 'mt' and mt_vocab:
                    val = [item for item in val if item in mt_vocab]
                counter[k].update(val)

        # Drop the none-using from memory but keep the last
        if (index < len(train_dataset_files) - 1):
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    _build_field_vocab(fields["tgt"], counter["tgt"],
                       max_size=tgt_vocab_size,
                       min_freq=tgt_words_min_frequency)

    if "mt" in fields.keys():
        _build_field_vocab(fields["mt"], counter["mt"],
                           max_size=mt_vocab_size,
                           min_freq=mt_words_min_frequency)

        # Merge the mt and target vocabularies.
        # `tgt_vocab_size` is ignored when sharing vocabularies
        logger.info(" * merging mt and tgt vocab...")
        merged_vocab = merge_vocabs(
            [fields["mt"].vocab, fields["tgt"].vocab],
            vocab_size=mt_vocab_size)
        fields["mt"].vocab = merged_vocab
        fields["tgt"].vocab = merged_vocab
        logger.info(" * mt vocab size: %d." % len(fields["mt"].vocab))

    if "char_mt" in fields.keys():
        # Add a Character Vocabulary
        _build_field_vocab(fields["char_mt"], counter["mt"],
                           counter["char_mt"],
                           max_size=mt_vocab_size,
                           min_freq=mt_words_min_frequency)
        logger.info(" * char mt vocab size: %d." %
                    len(fields["char_mt"].nesting_field.vocab))

    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

    if "char_tgt" in fields.keys():
        # Add a Character Vocabulary
        _build_field_vocab(fields["char_tgt"], counter["tgt"],
                           counter["char_tgt"],
                           max_size=tgt_vocab_size,
                           min_freq=tgt_words_min_frequency)
        logger.info(" * char tgt vocab size: %d." %
                    len(fields["char_tgt"].nesting_field.vocab))

    # All datasets have same num of n_tgt_features,
    # getting the last one is OK.
    for j in range(dataset.n_tgt_feats):
        key = "tgt_feat_" + str(j)
        _build_field_vocab(fields[key], counter[key])
        logger.info(" * %s vocab size: %d." % (key,
                                               len(fields[key].vocab)))

    if data_type == 'text':
        _build_field_vocab(fields["src"], counter["src"],
                           max_size=src_vocab_size,
                           min_freq=src_words_min_frequency)
        logger.info(" * src vocab size: %d." % len(fields["src"].vocab))

        if "char_src" in fields.keys():
            # Add a Character Vocabulary
            _build_field_vocab(fields["char_src"], counter["src"],
                               counter["char_src"],
                               max_size=src_vocab_size,
                               min_freq=src_words_min_frequency)
            logger.info(" * char src vocab size: %d." %
                        len(fields["char_src"].nesting_field.vocab))

        # All datasets have same num of n_src_features,
        # getting the last one is OK.
        for j in range(dataset.n_src_feats):
            key = "src_feat_" + str(j)
            _build_field_vocab(fields[key], counter[key])
            logger.info(" * %s vocab size: %d." %
                        (key, len(fields[key].vocab)))

        # Merge the input and output vocabularies.
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            logger.info(" * merging src and tgt vocab...")
            merged_vocab = merge_vocabs(
                [fields["src"].vocab, fields["tgt"].vocab],
                vocab_size=src_vocab_size)
            fields["src"].vocab = merged_vocab
            fields["tgt"].vocab = merged_vocab

    return fields


def _build_field_vocab_mono(field, counter, char_counter=None, **kwargs):
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.init_token,
                        field.eos_token]
        if tok is not None))
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)

    # Check if this field has an associated nested field for characters
    if hasattr(field, 'nesting_field'):
        # The current field has a nested character field, so we build
        # a vocabulary for the nested field as well
        char_specials = list(OrderedDict.fromkeys(
            tok for tok in [field.nesting_field.unk_token,
                            field.nesting_field.pad_token,
                            field.nesting_field.init_token,
                            field.nesting_field.eos_token
                            ]
            if tok is not None))
        # Make sure special tokens are not repeated
        for special in specials:
            if special not in char_specials:
                char_specials.append(special)
        # Create the vocab for the nested field
        field.nesting_field.vocab = field.nesting_field.vocab_cls(
                                char_counter,
                                specials=char_specials,
                                **kwargs)


def build_vocab_mono(train_dataset_files, fields,
                     tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    counter = {}
    for k in fields:
        counter[k] = Counter()

    # Load vocabulary
    tgt_vocab = None
    # if len(tgt_vocab_path) > 0:
    if tgt_vocab_path:
        tgt_vocab = set([])
        logger.info('Loading target vocab from %s' % tgt_vocab_path)
        assert os.path.exists(tgt_vocab_path), \
            'tgt vocab %s not found!' % tgt_vocab_path
        with open(tgt_vocab_path) as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue
                word = line.strip().split()[0]
                tgt_vocab.add(word)

    if "char_tgt" in fields.keys():
        n_chars = 256
        for idx in range(n_chars):
            # Create a counter for characters. By doing (n_chars - idx)
            # we are preserving the Unicode order of the characters by
            # giving the frequencies in descending order
            counter["char_tgt"][chr(idx)] = n_chars - idx

    for path in train_dataset_files:
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for k in fields:
                val = getattr(ex, k, None)
                if val is not None and not fields[k].sequential:
                    val = [val]
                elif val is not None and 'char' in k:
                    # ignore character fields, these were already
                    # initialized to be all unicode characters from 0 to 255.
                    continue
                elif k == 'tgt' and tgt_vocab:
                    val = [item for item in val if item in tgt_vocab]
                counter[k].update(val)

    _build_field_vocab_mono(fields["tgt"], counter["tgt"],
                            max_size=tgt_vocab_size,
                            min_freq=tgt_words_min_frequency)
    logger.info(" * tgt vocab size: %d." % len(fields["tgt"].vocab))

    if "char_tgt" in fields.keys():
        # Add a Character Vocabulary
        _build_field_vocab_mono(fields["char_tgt"], counter["tgt"],
                                counter["char_tgt"],
                                max_size=tgt_vocab_size,
                                min_freq=tgt_words_min_frequency)
        logger.info(" * char tgt vocab size: %d." %
                    len(fields["char_tgt"].nesting_field.vocab))

    return fields


def load_vocabulary(vocabulary_path, tag=""):
    """
    Loads a vocabulary from the given path.
    :param vocabulary_path: path to load vocabulary from
    :param tag: tag for vocabulary (only used for logging)
    :return: vocabulary or None if path is null
    """
    vocabulary = None
    if vocabulary_path:
        vocabulary = set([])
        logger.info("Loading {} vocabulary from {}".format(tag,
                                                           vocabulary_path))

        if not os.path.exists(vocabulary_path):
            raise RuntimeError(
                "{} vocabulary not found at {}!".format(tag, vocabulary_path))
        else:
            with open(vocabulary_path) as f:
                for line in f:
                    if len(line.strip()) == 0:
                        continue
                    word = line.strip().split()[0]
                    vocabulary.add(word)
    return vocabulary


class OrderedIterator(torchtext.data.Iterator):
    """ Ordered Iterator Class """

    def create_batches(self):
        """ Create batches """
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class LMIterator(OrderedIterator):

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a
                    # minibatch be sorted by decreasing order, which
                    # requires reversing relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                batch = torchtext.data.Batch(minibatch, self.dataset,
                                             self.device)
                batch.tgt = batch.tgt[1:-1].contiguous()
                if hasattr(batch, 'char_tgt'):
                    batch.char_tgt = batch.char_tgt[
                        :, 1:-1].contiguous()
                yield batch
            if not self.repeat:
                return


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """

    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train, lm_iter=False):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train
        self.use_lm_iterator = lm_iter
        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset.examples = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        if self.use_lm_iterator:
            return LMIterator(
                dataset=self.cur_dataset, batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device, train=self.is_train,
                sort=False, sort_within_batch=True,
                repeat=False)
        else:
            return OrderedIterator(
                dataset=self.cur_dataset, batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device, train=self.is_train,
                sort=False, sort_within_batch=True,
                repeat=False)


def build_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            """
            In token batching scheme, the number of sequences is limited
            such that the total number of src/tgt tokens (including padding)
            in a batch <= batch_size
            """
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)
    else:
        batch_size_fn = None

    if opt.gpu_ranks:
        device = "cuda"
    else:
        device = "cpu"

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train, opt.lm)


def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def _load_fields(dataset, data_type, opt, checkpoint, use_char=False,
                 get_ext_fields=False):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = load_fields_from_vocab(
            checkpoint['vocab'], data_type, use_char)
        if get_ext_fields:
            ext_fields = load_fields_from_vocab(
                torch.load(opt.data + '.vocab.pt'), data_type, use_char)
    else:
        fields = load_fields_from_vocab(
            torch.load(opt.data + '.vocab.pt'), data_type, use_char)

    if opt.pretrained_softmax_path:
        # if using a previously trained softmax from a LM,
        # the tgt field needs to be in sync with the pretrained LM
        lm_checkpoint = torch.load(
                            opt.pretrained_softmax_path,
                            map_location=lambda storage, loc: storage)
        lm_fields = load_fields_from_vocab(
            lm_checkpoint['vocab'], 'monotext', use_char)
        # Insert pad token in the LM field since LMs don't use padding
        lm_fields["tgt"].vocab.itos = [lm_fields["tgt"].vocab.itos[0]] +\
            [PAD_WORD] + lm_fields["tgt"].vocab.itos[1:]
        new_stoi = defaultdict(lambda: 0)
        new_stoi.update(
            {tok: i for i, tok in enumerate(lm_fields["tgt"].vocab.itos)})
        lm_fields["tgt"].vocab.stoi = new_stoi
        lm_fields['tgt'].pad_token = PAD_WORD
        # Add any token that is left from the current dataset
        lm_fields['tgt'].vocab.extend(fields['tgt'].vocab)
        # Replace the tgt field
        fields['tgt'] = lm_fields['tgt']

    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text':
        logger.info(' * vocabulary size. source = %d; target = %d' %
                    (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    elif data_type == 'monotext':
        logger.info(' * vocabulary size. target = %d' %
                    (len(fields['tgt'].vocab)))
    else:
        logger.info(' * vocabulary size. source = %d; mt = %d; target = %d' %
                    (len(fields['src'].vocab),
                     len(fields['mt'].vocab),
                     len(fields['tgt'].vocab)))

    if not get_ext_fields:
        ext_fields = None

    return fields, ext_fields


def _collect_report_features(fields):
    src_features = collect_features(fields, side='src')
    tgt_features = collect_features(fields, side='tgt')

    return src_features, tgt_features
