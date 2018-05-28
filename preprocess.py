#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import sys

import torch

import onmt.io
import onmt.opts


def check_existing_pt_files(opt):
    # We will use glob.glob() to find sharded {train|valid}.[0-9]*.pt
    # when training, so check to avoid tampering with existing pt files
    # or mixing them up.
    for t in ['train', 'valid', 'vocab']:
        pattern = opt.save_data + '.' + t + '*.pt'
        if glob.glob(pattern):
            sys.stderr.write("Please backup exisiting pt file: %s, "
                             "to avoid tampering!\n" % pattern)
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    onmt.opts.add_md_help_argument(parser)
    onmt.opts.preprocess_opts(parser)

    opt = parser.parse_args()

    invalid = not opt.train_src or not opt.valid_src

    if not opt.data_type == 'monotext' and invalid:
        sys.stderr.write("src paths (train_src and valid_src) are required,"
                         " unless using data_type option 'monotext'\n")
        sys.exit(1)

    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def build_save_text_dataset_in_shards(src_corpus, tgt_corpus, fields,
                                      corpus_type, opt, mt_corpus=None):
    '''
    Divide the big corpus into shards, and build dataset separately.
    This is currently only for data_type=='text'.

    The reason we do this is to avoid taking up too much memory due
    to sucking in a huge corpus file.

    To tackle this, we only read in part of the corpus file of size
    `max_shard_size`(actually it is multiples of 64 bytes that equals
    or is slightly larger than this size), and process it into dataset,
    then write it to disk along the way. By doing this, we only focus on
    part of the corpus at any moment, thus effectively reducing memory use.
    According to test, this method can reduce memory footprint by ~50%.

    Note! As we process along the shards, previous shards might still
    stay in memory, but since we are done with them, and no more
    reference to them, if there is memory tight situation, the OS could
    easily reclaim these memory.

    If `max_shard_size` is 0 or is larger than the corpus size, it is
    effectively preprocessed into one dataset, i.e. no sharding.

    NOTE! `max_shard_size` is measuring the input corpus size, not the
    output pt file size. So a shard pt file consists of examples of size
    2 * `max_shard_size`(source + target).
    '''

    data_type = 'text'

    corpus_size = os.path.getsize(tgt_corpus)
    if corpus_size > 10 * (1024**2) and opt.max_shard_size == 0:
        print("Warning. The corpus %s is larger than 10M bytes, you can "
              "set '-max_shard_size' to process it by small shards "
              "to use less memory." % tgt_corpus)

    if opt.max_shard_size != 0:
        print(' * divide corpus into shards and build dataset separately'
              '(shard_size = %d bytes).' % opt.max_shard_size)

    ret_list = []
    if src_corpus is not None:
        src_iter = onmt.io.ShardedTextCorpusIterator(
            src_corpus, opt.src_seq_length_trunc,
            "src", opt.max_shard_size)
        src_num_features = src_iter.num_feats
    else:
        src_iter = None
        src_num_features = 0
        data_type = 'monotext'

    tgt_iter = onmt.io.ShardedTextCorpusIterator(
        tgt_corpus, opt.tgt_seq_length_trunc,
        "tgt", opt.max_shard_size,
        assoc_iter=src_iter)

    # In case we are doing the APE task
    if mt_corpus is not None:
        mt_iter = onmt.io.ShardedTextCorpusIterator(
            mt_corpus, opt.mt_seq_length_trunc,
            "mt", opt.max_shard_size,
            assoc_iter=src_iter)
        mt_num_features = mt_iter.num_feats
    else:
        mt_iter = None
        mt_num_features = 0

    index = 0
    while_end_condition = False
    while not while_end_condition:
        index += 1
        dataset = onmt.io.TextDataset(
            data_type, fields, src_iter, tgt_iter,
            mt_iter, mt_num_features,
            src_num_features, tgt_iter.num_feats,
            src_seq_length=opt.src_seq_length,
            tgt_seq_length=opt.tgt_seq_length,
            mt_seq_length=opt.mt_seq_length,
            dynamic_dict=opt.dynamic_dict)

        # We save fields in vocab.pt seperately, so make it empty.
        dataset.fields = []

        pt_file = "{:s}.{:s}.{:d}.pt".format(
            opt.save_data, corpus_type, index)
        print(" * saving %s data shard to %s." % (corpus_type, pt_file))
        torch.save(dataset, pt_file)

        ret_list.append(pt_file)

        if data_type == 'text':
            while_end_condition = src_iter.hit_end()
        else:
            while_end_condition = tgt_iter.hit_end()

    return ret_list


def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
        if opt.train_mt:
            mt_corpus = opt.train_mt
        else:
            mt_corpus = None
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt
        if opt.valid_mt:
            mt_corpus = opt.valid_mt
        else:
            mt_corpus = None

    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    if opt.data_type == 'text' or opt.data_type == 'monotext':
        return build_save_text_dataset_in_shards(
            src_corpus, tgt_corpus, fields,
            corpus_type, opt, mt_corpus)

    # For data_type == 'img' or 'audio', currently we don't do
    # preprocess sharding. We only build a monolithic dataset.
    # But since the interfaces are uniform, it would be not hard
    # to do this should users need this feature.
    dataset = onmt.io.build_dataset(
        fields, opt.data_type, src_corpus, tgt_corpus,
        src_dir=opt.src_dir,
        src_seq_length=opt.src_seq_length,
        tgt_seq_length=opt.tgt_seq_length,
        src_seq_length_trunc=opt.src_seq_length_trunc,
        tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
        dynamic_dict=opt.dynamic_dict,
        sample_rate=opt.sample_rate,
        window_size=opt.window_size,
        window_stride=opt.window_stride,
        window=opt.window)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    print(" * saving %s dataset to %s." % (corpus_type, pt_file))
    torch.save(dataset, pt_file)

    return [pt_file]


def build_save_vocab(train_dataset, fields, opt):
    fields = onmt.io.build_vocab(train_dataset, fields, opt.data_type,
                                 opt.share_vocab,
                                 opt.src_vocab,
                                 opt.src_vocab_size,
                                 opt.src_words_min_frequency,
                                 opt.tgt_vocab,
                                 opt.tgt_vocab_size,
                                 opt.tgt_words_min_frequency,
                                 opt.mt_vocab,
                                 opt.mt_vocab_size,
                                 opt.mt_words_min_frequency,
                                 opt.n_chars)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    torch.save(onmt.io.save_fields_to_vocab(fields), vocab_file)


def main():
    opt = parse_args()

    print("Extracting features...")
    src_nfeats = onmt.io.get_num_features(opt.data_type, opt.train_src, 'src')
    if opt.train_mt:
        mt_nfeats = onmt.io.get_num_features(opt.data_type,
                                             opt.train_tgt, 'mt')
    else:
        mt_nfeats = None
    tgt_nfeats = onmt.io.get_num_features(opt.data_type, opt.train_tgt, 'tgt')
    print(" * number of source features: %d." % src_nfeats)
    if opt.train_mt:
        print(" * number of mt features: %d." % mt_nfeats)
    print(" * number of target features: %d." % tgt_nfeats)

    print("Building `Fields` object...")
    fields = onmt.io.get_fields(opt.data_type, src_nfeats, tgt_nfeats,
                                opt.use_char, mt_nfeats)

    print("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    print("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)

    print("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)


if __name__ == "__main__":
    main()
