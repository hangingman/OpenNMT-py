#!/usr/bin/env python
from __future__ import division, unicode_literals
import argparse

from onmt.translate.Translator import make_translator

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()


def _report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def _report_bleu():
    import subprocess
    print()
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s < %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    res = subprocess.check_output(
        "python tools/test_rouge.py -r %s -c %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    # Test data
    # TODO: if the model was trained without fertility but tested with
    # fertility, need to use opt instead of model_opt below.
    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.tgt,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False,
                                 fertility_type=model_opt.fertility_type,
                                 fertility_value=model_opt.fertility)

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_within_batch=True, shuffle=False)

    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta)
    translator = onmt.translate.Translator(
        model, fields,
        beam_size=opt.beam_size,
        n_best=opt.n_best,
        global_scorer=scorer,
        max_length=opt.max_length,
        copy_attn=model_opt.copy_attn,
        attn_transform=model_opt.attn_transform,
        c_attn=model_opt.c_attn,
        cuda=opt.cuda,
        beam_trace=opt.dump_beam != "",
        min_length=opt.min_length)
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, opt.tgt)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    attn_matrices = []
    gold_attn_matrices = []

    for batch in data_iter:
        batch_data = translator.translate_batch(batch, data)
        attn_matrices.append(batch_data['attention'])
        if opt.tgt:
            gold_attn_matrices.append(batch_data['gold_attention'])
        translations = builder.from_batch(batch_data)

        for trans in translations:
            pred_score_total += trans.pred_scores[0]
            pred_words_total += len(trans.pred_sents[0])
            if opt.tgt:
                gold_score_total += trans.gold_score
                gold_words_total += len(trans.gold_sent)

            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

            if opt.verbose:
                sent_number = next(counter)
                output = trans.log(sent_number)
                os.write(1, output.encode('utf-8'))
=======
import onmt.opts
>>>>>>> 0ecec8b4c16fdec7d8ce2646a0ea47ab6535d308


def main(opt):
    translator = make_translator(opt, report_score=True)
    translator.translate(opt.src_dir, opt.src, opt.tgt,
                         opt.batch_size, opt.attn_debug)

    if opt.dump_attn:
        attn_matrices = [a[0][0].cpu().numpy() for a in attn_matrices]
        gold_attn_matrices = [a['std'][:,0,:].data.cpu().numpy()
                              for a in gold_attn_matrices]
        import pickle
        pickle.dump({'pred': attn_matrices, 'gold': gold_attn_matrices},
                   open('attn_matrices_' + model_opt.attn_transform + '.out',
                        'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)
