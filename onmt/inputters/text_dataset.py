# -*- coding: utf-8 -*-
"""Define word-based embedders."""

from collections import Counter
from itertools import chain
import io
import codecs
import sys

import torch
import torchtext

from onmt.inputters.dataset_base import (DatasetBase, UNK_WORD,
                                         PAD_WORD, BOS_WORD, EOS_WORD,
                                         BOW_CHAR, EOW_CHAR)
from onmt.utils.misc import aeq


class TextDataset(DatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def __init__(self, fields, src_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_tgt_feats=0,
                 src_seq_length=0, tgt_seq_length=0,
                 dynamic_dict=True, use_filter_pred=True):

        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.src_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_tgt_feats = num_tgt_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                             zip(src_examples_iter, tgt_examples_iter))
        else:
            examples_iter = src_examples_iter

        if dynamic_dict:
            examples_iter = self._dynamic_dict(examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            src_size += len(example.src)
            out_examples.append(example)

        print("average src size", src_size / len(out_examples),
              len(out_examples))

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
                and 0 < len(example.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        # Default to a balanced sort, prioritizing tgt len match.
        # TODO: make this configurable.
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices.data[b]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)
        return scores

    @staticmethod
    def make_text_examples_nfeats_tpl(text_iter, text_path, truncate, side):
        """
        Args:
            text_iter(iterator): an iterator (or None) that we can loop over
                to read examples.
                It may be an openned file, a string list etc...
            text_path(str): path to file or None
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ['src', 'tgt']

        if text_iter is None:
            if text_path is not None:
                text_iter = TextDataset.make_text_iterator_from_file(text_path)
            else:
                return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            TextDataset.make_examples(text_iter, truncate, side)

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)

    @staticmethod
    def make_examples(text_iter, truncate, side):
        """
        Args:
            text_iter (iterator): iterator of text sequences
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            (word, features, nfeat) triples for each line.
        """
        for i, line in enumerate(text_iter):
            line = line.strip().split()
            if truncate:
                line = line[:truncate]

            words, feats, n_feats = \
                TextDataset.extract_text_features(line)
            char_side = "char_" + side
            example_dict = {side: words, "indices": i, char_side: words}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))
            yield example_dict, n_feats

    @staticmethod
    def make_text_iterator_from_file(path):
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    @staticmethod
    def get_fields(n_src_features, n_tgt_features, use_char=False):
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.
            use_char (bool): boolean to decide if character fields are
                necessary.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_" + str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        if use_char:
            # Create character related fields
            nesting_field_src = torchtext.data.Field(tokenize=list,
                                                     pad_token=PAD_WORD,
                                                     init_token=BOW_CHAR,
                                                     eos_token=EOW_CHAR,
                                                     fix_length=50)

            fields["char_src"] = torchtext.data.NestedField(
                                    nesting_field_src,
                                    init_token=BOS_WORD,
                                    eos_token=EOS_WORD,
                                    pad_token=PAD_WORD)

            nesting_field_tgt = torchtext.data.Field(tokenize=list,
                                                     pad_token=PAD_WORD,
                                                     init_token=BOW_CHAR,
                                                     eos_token=EOW_CHAR,
                                                     fix_length=50)

            fields["char_tgt"] = torchtext.data.NestedField(
                                    nesting_field_tgt,
                                    init_token=BOS_WORD,
                                    eos_token=EOS_WORD,
                                    pad_token=PAD_WORD)

        def make_src(data, vocab):
            """ ? """
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)

        def make_tgt(data, vocab):
            """ ? """
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        Peek one line and get number of features of it.
        (All lines must have same number of features).
        For text corpus, both sides are in text form, thus
        it works the same.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        with codecs.open(corpus_file, "r", "utf-8") as cf:
            f_line = cf.readline().strip().split()
            _, _, num_feats = TextDataset.extract_text_features(f_line)

        return num_feats

    # Below are helper functions for intra-class use only.
    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            src = example["src"]
            src_vocab = torchtext.vocab.Vocab(Counter(src),
                                              specials=[UNK_WORD, PAD_WORD])
            self.src_vocabs.append(src_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            src_map = torch.LongTensor([src_vocab.stoi[w] for w in src])
            example["src_map"] = src_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                    [0] + [src_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example


class MonotextDataset(DatasetBase):
    """ Dataset for data_type=='text'

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'char_text' and 'text'.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            tgt_seq_length (int): maximum target sequence length.
    """

    def __init__(self, fields, tgt_examples_iter, bptt_len):

        self.data_type = 'monotext'

        # Each element of an example is a dictionary
        examples_iter = (self._join_dicts(tgt) for tgt in
                         tgt_examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        out_examples = []
        rollover_tokens = dict()
        n_tokens = 0

        for ex_values in example_values:
            examples, rollover_tokens = self._construct_example_fromlist(
                ex_values, out_fields, bptt_len, rollover_tokens)
            if len(examples) > 0:
                for example in examples:
                    n_tokens += len(example.tgt)
                    out_examples.append(example)

        out_examples_with_indices = []
        for i, example in enumerate(out_examples):
            setattr(example, 'indices', i)
            out_examples_with_indices.append(example)
        out_examples = out_examples_with_indices
        print("Number of tokens in shard: ", n_tokens)

        super(MonotextDataset, self).__init__(out_examples, out_fields)

    def _construct_example_fromlist(self, data, fields,
                                    bptt_len, rollover_tokens=None):
        """
        Args:
            data: the data to be set as the value of the attributes of
                the to-be-created `Example`, associating with respective
                `Field` objects with same key.
            fields: a dict of `torchtext.data.Field` objects. The keys
                are attributes of the to-be-created `Example`.

        Returns:
            the created `Example` object.
        """

        ex = torchtext.data.Example()
        rollover_exs = []

        for (name, field), val in zip(fields, data):
            if field is None:
                setattr(ex, name, val)
                continue
            elif 'char' in name:
                preprocessed = self._construct_char_line(val, field)
            elif name == 'indices':
                continue
            else:
                preprocessed = tuple([field.init_token]) +\
                    field.preprocess(val) +\
                    tuple([field.eos_token])

            if name in rollover_tokens.keys() and \
                    len(rollover_tokens[name]) > 0:
                preprocessed = rollover_tokens[name] + preprocessed

            new_text = preprocessed[:bptt_len]

            if len(new_text) == bptt_len:
                setattr(ex, name, new_text)
                rollover_tokens[name] = preprocessed[bptt_len:]
            else:
                rollover_tokens[name] = preprocessed

            i = 0
            while len(rollover_tokens[name]) >= bptt_len:

                if len(rollover_exs) == i:
                    rollover_exs.append(torchtext.data.Example())

                rollover_ex = rollover_exs[i]

                new_text = rollover_tokens[name][:bptt_len]

                if len(new_text) == bptt_len:
                    setattr(rollover_ex, name, new_text)
                    rollover_tokens[name] = rollover_tokens[name][bptt_len:]
                    rollover_exs[i] = rollover_ex
                    i += 1

            if len(rollover_tokens[name]) == 0:
                rollover_tokens[name] = []

        if not hasattr(ex, 'tgt'):
            exs = []
        else:
            exs = [ex]

        for rollover_ex in rollover_exs:
            if hasattr(rollover_ex, 'tgt'):
                exs.append(rollover_ex)

        return exs, rollover_tokens

    def _construct_char_line(self, val, field):
        preprocessed = field.preprocess(val)
        # if we are preprocessing an example with a character
        # field, we want to ensure all characters are encoded
        # in utf-8. This way, we will turn characters that have
        # a unicode > 255 in several characters with
        # unicode < 256.
        # This ensures we do not have unknown characters
        # in character-based embeddings.
        new_prepr = [[field.init_token]]
        # new_prepr = []
        for word in preprocessed:
            new_word = []
            for char in word:
                new_chr = char.encode('utf-8', 'ignore')
                # check if character was split into
                # several characters
                for code in new_chr:
                    new_word.append(chr(code))
            new_prepr.append(new_word)
        preprocessed = new_prepr + [[field.eos_token]]
        return preprocessed

    def sort_key(self, ex):
        """ Sort using length of source sentences. """
        return len(ex.tgt)

    @staticmethod
    def make_text_examples_nfeats_tpl(text_iter, text_path, truncate, side):
        """
        Args:
            text_iter(iterator): an iterator (or None) that we can loop over
                to read examples.
                It may be an openned file, a string list etc...
            text_path(str): path to file or None
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ['src', 'tgt']

        if text_iter is None:
            if text_path is not None:
                text_iter = TextDataset.make_text_iterator_from_file(text_path)
            else:
                return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            TextDataset.make_examples(text_iter, truncate, side)

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)

    @staticmethod
    def make_examples(text_iter, truncate, side):
        """
        Args:
            text_iter (iterator): iterator of text sequences
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            (word, features, nfeat) triples for each line.
        """
        for i, line in enumerate(text_iter):
            line = line.strip().split()
            if truncate:
                line = line[:truncate]

            words, feats, n_feats = \
                TextDataset.extract_text_features(line)

            example_dict = {side: words, "indices": i}
            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))
            yield example_dict, n_feats

    @staticmethod
    def make_text_iterator_from_file(path):
        with codecs.open(path, "r", "utf-8") as corpus_file:
            for line in corpus_file:
                yield line

    @staticmethod
    def get_fields(n_tgt_features, use_char=False):
        """
        Args:
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.
            use_char (bool): boolean to decide if character fields are
                necessary.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD)

        if use_char:
            # Create character related fields
            nesting_field_tgt = torchtext.data.Field(tokenize=list,
                                                     pad_token=PAD_WORD,
                                                     init_token=BOW_CHAR,
                                                     eos_token=EOW_CHAR,
                                                     fix_length=50)

            fields["char_tgt"] = torchtext.data.NestedField(
                                    nesting_field_tgt,
                                    init_token=BOS_WORD,
                                    eos_token=EOS_WORD)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        return fields


class APETextDataset(TextDataset):
    """ Dataset for APE task.

        Build `Example` objects, `Field` objects, and filter_pred function
        from text corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
                Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            num_src_feats (int): number of source side features.
            num_tgt_feats (int): number of target side features.
            src_seq_length (int): maximum source sequence length.
            tgt_seq_length (int): maximum target sequence length.
            dynamic_dict (bool): create dynamic dictionaries?
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def __init__(self, fields,
                 src_examples_iter, mt_examples_iter, tgt_examples_iter,
                 num_src_feats=0, num_mt_feats=0, num_tgt_feats=0,
                 src_seq_length=0, mt_seq_length=0, tgt_seq_length=0,
                 dynamic_dict=True, use_filter_pred=True):

        self.data_type = 'text'

        # self.src_vocabs: mutated in dynamic_dict, used in
        # collapse_copy_scores and in Translator.py
        self.mt_vocabs = []

        self.n_src_feats = num_src_feats
        self.n_mt_feats = num_mt_feats
        self.n_tgt_feats = num_tgt_feats

        # Each element of an example is a dictionary whose keys represents
        # at minimum the src tokens and their indices and potentially also
        # the src and tgt features and alignment information.
        if src_examples_iter is not None and tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src, mt, tgt) for src, mt, tgt in
                             zip(src_examples_iter, mt_examples_iter,
                                 tgt_examples_iter))
        elif tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(mt, tgt) for mt, tgt in
                             zip(mt_examples_iter,
                                 tgt_examples_iter))
        else:
            examples_iter = (self._join_dicts(src, mt) for src, mt in
                             zip(src_examples_iter, mt_examples_iter))

        examples_iter = self._dynamic_dict(examples_iter)

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)

        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        src_size = 0

        out_examples = []
        for ex_values in example_values:
            example = self._construct_example_fromlist(
                ex_values, out_fields)
            src_size += len(example.src)
            out_examples.append(example)

        print("average src size", src_size / len(out_examples),
              len(out_examples))

        def filter_pred(example):
            return 0 < len(example.src) <= src_seq_length \
                and 0 < len(example.mt) <= mt_seq_length \
                and 0 < len(example.tgt) <= tgt_seq_length

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(TextDataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    @staticmethod
    def get_fields(n_src_features, n_mt_features, n_tgt_features,
                   use_char=False):
        """
        Args:
            n_src_features (int): the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of mt features to
                create `torchtext.data.Field` for.
            n_tgt_features (int): the number of target features to
                create `torchtext.data.Field` for.
            use_char (bool): boolean to decide if character fields are
                necessary.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        fields["src"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["mt"] = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)

        for j in range(n_mt_features):
            fields["mt_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_" + str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        if use_char:
            # Create character related fields
            nesting_field_src = torchtext.data.Field(tokenize=list,
                                                     pad_token=PAD_WORD,
                                                     init_token=BOW_CHAR,
                                                     eos_token=EOW_CHAR,
                                                     fix_length=50)

            fields["char_src"] = torchtext.data.NestedField(
                                    nesting_field_src,
                                    init_token=BOS_WORD,
                                    eos_token=EOS_WORD,
                                    pad_token=PAD_WORD)

            nesting_field_mt = torchtext.data.Field(tokenize=list,
                                                    pad_token=PAD_WORD,
                                                    init_token=BOW_CHAR,
                                                    eos_token=EOW_CHAR,
                                                    fix_length=50)

            fields["char_mt"] = torchtext.data.NestedField(
                                    nesting_field_mt,
                                    init_token=BOS_WORD,
                                    eos_token=EOS_WORD,
                                    pad_token=PAD_WORD)

            nesting_field_tgt = torchtext.data.Field(tokenize=list,
                                                     pad_token=PAD_WORD,
                                                     init_token=BOW_CHAR,
                                                     eos_token=EOW_CHAR,
                                                     fix_length=50)

            fields["char_tgt"] = torchtext.data.NestedField(
                                    nesting_field_tgt,
                                    init_token=BOS_WORD,
                                    eos_token=EOS_WORD,
                                    pad_token=PAD_WORD)

        def make_src(data, vocab):
            """ ? """
            src_size = max([t.size(0) for t in data])
            src_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(src_size, len(data), src_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["src_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)

        def make_mt(data, vocab):
            """ ? """
            mt_size = max([t.size(0) for t in data])
            mt_vocab_size = max([t.max() for t in data]) + 1
            alignment = torch.zeros(mt_size, len(data), mt_vocab_size)
            for i, sent in enumerate(data):
                for j, t in enumerate(sent):
                    alignment[j, i, t] = 1
            return alignment

        fields["mt_map"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_mt, sequential=False)

        def make_tgt(data, vocab):
            """ ? """
            tgt_size = max([t.size(0) for t in data])
            alignment = torch.zeros(tgt_size, len(data)).long()
            for i, sent in enumerate(data):
                alignment[:sent.size(0), i] = sent
            return alignment

        fields["alignment"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        return fields

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, mt_vocabs):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambigious.
        """
        offset = len(tgt_vocab)
        for b in range(batch.batch_size):
            blank = []
            fill = []
            index = batch.indices.data[b]
            src_vocab = mt_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                scores[:, b].index_add_(1, fill,
                                        scores[:, b].index_select(1, blank))
                scores[:, b].index_fill_(1, blank, 1e-10)
        return scores

    @staticmethod
    def make_text_examples_nfeats_tpl(text_iter, text_path, truncate, side):
        """
        Args:
            text_iter(iterator): an iterator (or None) that we can loop over
                to read examples.
                It may be an openned file, a string list etc...
            text_path(str): path to file or None
            path (str): location of a src or tgt file.
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Returns:
            (example_dict iterator, num_feats) tuple.
        """
        assert side in ['src', 'mt', 'tgt']

        if text_iter is None:
            if text_path is not None:
                text_iter = TextDataset.make_text_iterator_from_file(text_path)
            else:
                return (None, 0)

        # All examples have same number of features, so we peek first one
        # to get the num_feats.
        examples_nfeats_iter = \
            TextDataset.make_examples(text_iter, truncate, side)

        first_ex = next(examples_nfeats_iter)
        num_feats = first_ex[1]

        # Chain back the first element - we only want to peek it.
        examples_nfeats_iter = chain([first_ex], examples_nfeats_iter)
        examples_iter = (ex for ex, nfeats in examples_nfeats_iter)

        return (examples_iter, num_feats)

    # Below are helper functions for intra-class use only.
    def _dynamic_dict(self, examples_iter):
        for example in examples_iter:
            mt = example["mt"]
            mt_vocab = torchtext.vocab.Vocab(Counter(mt),
                                             specials=[UNK_WORD, PAD_WORD])
            self.mt_vocabs.append(mt_vocab)
            # Mapping source tokens to indices in the dynamic dict.
            mt_map = torch.LongTensor([mt_vocab.stoi[w] for w in mt])
            example["mt_map"] = mt_map

            if "tgt" in example:
                tgt = example["tgt"]
                mask = torch.LongTensor(
                    [0] + [mt_vocab.stoi[w] for w in tgt] + [0])
                example["alignment"] = mask
            yield example


class ShardedTextCorpusIterator(object):
    """
    This is the iterator for text corpus, used for sharding large text
    corpus into small shards, to avoid hogging memory.

    Inside this iterator, it automatically divides the corpus file into
    shards of size `shard_size`. Then, for each shard, it processes
    into (example_dict, n_features) tuples when iterates.
    """

    def __init__(self, corpus_path, line_truncate, side, shard_size,
                 assoc_iter=None):
        """
        Args:
            corpus_path: the corpus file path.
            line_truncate: the maximum length of a line to read.
                            0 for unlimited.
            side: "src" or "tgt".
            shard_size: the shard size, 0 means not sharding the file.
            assoc_iter: if not None, it is the associate iterator that
                        this iterator should align its step with.
        """
        try:
            # The codecs module seems to have bugs with seek()/tell(),
            # so we use io.open().
            self.corpus = io.open(corpus_path, "r", encoding="utf-8")
        except IOError:
            sys.stderr.write("Failed to open corpus file: %s" % corpus_path)
            sys.exit(1)

        self.line_truncate = line_truncate
        self.side = side
        self.shard_size = shard_size
        self.assoc_iter = assoc_iter
        self.last_pos = 0
        self.line_index = -1
        self.eof = False

    def __iter__(self):
        """
        Iterator of (example_dict, nfeats).
        On each call, it iterates over as many (example_dict, nfeats) tuples
        until this shard's size equals to or approximates `self.shard_size`.
        """
        iteration_index = -1
        if self.assoc_iter is not None:
            # We have associate iterator, just yields tuples
            # util we run parallel with it.
            while self.line_index < self.assoc_iter.line_index:
                line = self.corpus.readline()
                if line == '':
                    raise AssertionError(
                        "Two corpuses must have same number of lines!")

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

            if self.assoc_iter.eof:
                self.eof = True
                self.corpus.close()
        else:
            # Yield tuples util this shard's size reaches the threshold.
            self.corpus.seek(self.last_pos)
            while True:
                if self.shard_size != 0 and self.line_index % 64 == 0:
                    # This part of check is time consuming on Py2 (but
                    # it is quite fast on Py3, weird!). So we don't bother
                    # to check for very line. Instead we chekc every 64
                    # lines. Thus we are not dividing exactly per
                    # `shard_size`, but it is not too much difference.
                    cur_pos = self.corpus.tell()
                    if cur_pos >= self.last_pos + self.shard_size:
                        self.last_pos = cur_pos
                        raise StopIteration

                line = self.corpus.readline()
                if line == '':
                    self.eof = True
                    self.corpus.close()
                    raise StopIteration

                self.line_index += 1
                iteration_index += 1
                yield self._example_dict_iter(line, iteration_index)

    def hit_end(self):
        """ ? """
        return self.eof

    @property
    def num_feats(self):
        """
        We peek the first line and seek back to
        the beginning of the file.
        """
        saved_pos = self.corpus.tell()

        line = self.corpus.readline().split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        _, _, self.n_feats = TextDataset.extract_text_features(line)

        self.corpus.seek(saved_pos)

        return self.n_feats

    def _example_dict_iter(self, line, index):
        line = line.split()
        if self.line_truncate:
            line = line[:self.line_truncate]
        words, feats, n_feats = TextDataset.extract_text_features(line)
        char_side = "char_" + self.side
        example_dict = {self.side: words, "indices": index, char_side: words}
        if feats:
            # All examples must have same number of features.
            aeq(self.n_feats, n_feats)

            prefix = self.side + "_feat_"
            example_dict.update((prefix + str(j), f)
                                for j, f in enumerate(feats))

        return example_dict
