import argparse
import torch
import codecs
import os
import math

from torch.autograd import Variable
from itertools import count

import onmt.ModelConstructor
import onmt.translate.Beam
import onmt.io
import onmt.opts

# ADDED -------------------------------
import pdb
import pickle
from collections import defaultdict
import numpy as np
import time
import itertools
#from gensim.models import FastText
# END ---------------------------------

def make_translator(opt, report_score=True, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w', 'utf-8')

    if opt.gpu > -1:
        torch.cuda.set_device(opt.gpu)

    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    # ADDED --------------------------------------------------------------
    extend_model = False
    #id_method = 'unk'
    id_method = 'pred'

    def _predict(x, model_dict):

        nr_layers = len(model_dict)//2
        
        for i in range(1, nr_layers+1):
            w = model_dict['fc'+str(i)+'.weight']
            b = model_dict['fc'+str(i)+'.bias']
            x = torch.matmul(w, x) + b
    
        return x
    
    if extend_model:
        # Get the fields for the in-domain model
        base_path = "/mnt/data/de-en-md-shr/base/"
        id_model = base_path + "de-en-md-shr_acc_79.67_ppl_2.58_e15.pt"
        id_check = torch.load(id_model,
                              map_location=lambda storage, loc: storage)
        id_fields = onmt.io.load_fields_from_vocab(id_check['vocab'],
                                                   data_type='text')
        
        id_vocab = id_fields['tgt'].vocab
        ge_vocab = fields['tgt'].vocab
   
        init_index = len(ge_vocab.itos)
        added = 0

        print("Updating vocab ...")
        for idw in id_vocab.itos:
            if idw not in ge_vocab.itos:
                fields['tgt'].vocab.itos.append(idw)
                fields['tgt'].vocab.stoi[idw] = init_index + added
                added += 1
        print("Vocab updated. Added {} new words.".format(added))

        print("Updating decoder and embeddings ...")
        id_w_mx = torch.empty(added, 
                              model.generator[0].weight.shape[1],
                              dtype=torch.float)

        id_b_mx = torch.empty(added,
                              dtype=torch.float)

        if id_method == 'unk':
            unk_w = model.generator[0].weight[0]
            unk_b = model.generator[0].bias[0]
            
            id_w_mx = unk_w.repeat([added, 1])
            id_b_mx = unk_b.repeat([added])

        elif id_method == 'pred':
            
            # Define the paths for now
            base_path = '/home/ubuntu/NMT-Code/attention_comparison/thesis/guided_nmt/embed_map'
            weight_model_path = base_path + '/model_weights.ckpt'
            bias_model_path = base_path + '/model_bias.ckpt'
            ft_model_path = '/mnt/ft/de-en/cc.de.300.bin'
            
            # Load the necessary models
            w_model = torch.load(weight_model_path)
            b_model = torch.load(bias_model_path)
            print("Loading ft model ...")
            ft_model = FastText.load_fasttext_format(ft_model_path)
            print("Finished loading the model")

            # Define unk weight and bias values
            unk_w = model.generator[0].weight[0]
            unk_b = model.generator[0].bias[0]

            for i in range(added):
                try:
                    wv_ = torch.from_numpy(ft_model.wv[fields['tgt'].vocab.itos[init_index+i]])
                    id_w_mx[i] = _predict(wv_, w_model)
                    id_b_mx[i] = _predict(wv_, b_model)
                except:
                    id_w_mx[i] = unk_w
                    id_b_mx[i] = unk_b

        print("Previous shape: ", model.generator[0].weight.shape)
        model.generator[0].weight.data = torch.cat((model.generator[0].weight, 
                                                    id_w_mx), 
                                                   0)
        model.generator[0].bias.data = torch.cat((model.generator[0].bias,
                                                  id_b_mx),
                                                 0)
        model.decoder.embeddings.word_lut.weight = model.generator[0].weight
        print("Current shape: ", model.generator[0].weight.shape)
        print("Finished updating decoder and embeddings")

    # END ----------------------------------------------------------------

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.min_attention,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam" , "dump_attn",
                        "data_type", "replace_unk", "gpu", "verbose",
                        "use_guided", "tp_path", "guided_n_max",
                        "guided_1_weight", "guided_n_weight",
                        "guided_correct_ngrams", "guided_correct_1grams",
                        "extend_with_tp", "extend_1_weight", "extend_n_weight",
                        "extend_pred"]}

    translator = Translator(model, model_opt, fields, global_scorer=scorer,
                            out_file=out_file, report_score=report_score,
                            copy_attn=model_opt.copy_attn, **kwargs)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """

    def __init__(self,
                 model,
                 model_opt,
                 fields,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 attn_transform="softmax", 
                 c_attn=0.0,
                 global_scorer=None,
                 copy_attn=False,
                 gpu=False,
                 dump_beam="",
                 dump_attn="",
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 sample_rate='16000',
                 window_size=.02,
                 window_stride=.01,
                 window='hamming',
                 use_filter_pred=False,
                 data_type="text",
                 replace_unk=False,
                 report_score=True,
                 report_bleu=False,
                 report_rouge=False,
                 verbose=False,
                 out_file=None,
                 use_guided=True,
                 tp_path="",
                 guided_n_max=4,
                 guided_1_weight=1.0,
                 guided_n_weight=1.0,
                 guided_correct_ngrams=False,
                 guided_correct_1grams=False,
                 extend_with_tp=False,
                 extend_1_weight=1.0,
                 extend_n_weight=1.0,
                 extend_pred=None):

        self.gpu = gpu
        self.cuda = gpu > -1


        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.attn_transform = attn_transform
        self.c_attn = c_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.dump_attn = dump_attn
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.use_filter_pred = use_filter_pred
        self.replace_unk = replace_unk
        self.data_type = data_type
        self.verbose = verbose
        self.out_file = out_file
        self.report_score = report_score
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.use_guided= use_guided
        self.tp_path = tp_path
        self.guided_n_max = guided_n_max
        self.guided_1_weight = guided_1_weight
        self.guided_n_weight = guided_n_weight
        self.guided_correct_ngrams = guided_correct_ngrams
        self.guided_correct_1grams = guided_correct_1grams
        self.extend_with_tp = extend_with_tp
        self.extend_1_weight = extend_1_weight
        self.extend_n_weight = extend_n_weight
        self.extend_pred = extend_pred

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate(self, src_dir, src_path, tgt_path,
                  batch_size, attn_debug=False):
        data = onmt.io.build_dataset(self.fields,
                                     self.data_type,
                                     src_path,
                                     tgt_path,
                                     src_dir=src_dir,
                                     sample_rate=self.sample_rate,
                                     window_size=self.window_size,
                                     window_stride=self.window_stride,
                                     window=self.window,
                                     use_filter_pred=self.use_filter_pred)

        data_iter = onmt.io.OrderedIterator(
            dataset=data, device=self.gpu,
            batch_size=batch_size, train=False, sort=False,
            sort_within_batch=True, shuffle=False)

        builder = onmt.translate.TranslationBuilder(
            data, self.fields,
            self.n_best, self.replace_unk, tgt_path)

        # ADDED --------------------------------------------------------------
        # Load the translation pieces list
        if self.use_guided:
            translation_pieces = pickle.load(open(self.tp_path, 'rb'), 
                                             encoding='latin1')

        if self.extend_pred:
            ft_model_path = '/mnt/ft/wiki.en.bin'
            base_path = '/home/ubuntu/NMT-Code/attention_comparison/thesis/guided_nmt/embed_map'
            
            print("Loading ft model ...")
            ft_model = FastText.load_fasttext_format(ft_model_path)
            print("Finished loading the model")

            if self.extend_pred == "ls":
                weight_model_path = base_path + "/ls_embed_mx.ckpt"
                bias_model_path = base_path + "/ls_bias_mx.ckpt"

            elif self.extend_pred == "mlp": 
                weight_model_path = base_path + '/model_weights.ckpt'
                bias_model_path = base_path + '/model_bias.ckpt'
                
            # Load the necessary models
            w_model = torch.load(weight_model_path)
            b_model = torch.load(bias_model_path)

            # Build the list that is passed to translation_batch
            models = [ft_model, w_model, b_model]

        tot_time = 0
        # END ----------------------------------------------------------------

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        attn_matrices = []
        gold_attn_matrices = []

        all_scores = []

        # ADDED -----------------------------------------------------------------
        # Target vocab init size to properly weight extend_pred subwords
        orig_vocab_ix = len(self.fields["tgt"].vocab.itos)
        print("Initial vocab size: ", orig_vocab_ix) 
        # END -------------------------------------------------------------------
        for ix, batch in enumerate(data_iter):
            # ADDED --------------------------------------------------------------
            start_time = time.time()

            if self.use_guided:
                if self.extend_with_tp:
                    if self.extend_pred:
                        batch_data = self.translate_batch(batch, data, 
                                                          translation_pieces, models, orig_vocab_ix)
                    else:
                        batch_data = self.translate_batch(batch, data, 
                                                          translation_pieces, list(), orig_vocab_ix)
                else:
                    batch_data = self.translate_batch(batch, data, 
                                                      translation_pieces, list(), 0)
    
            else:
                batch_data = self.translate_batch(batch, data, dict(), list(), 0)
            
            # END ----------------------------------------------------------------
            attn_matrices.append(batch_data['attention'])
            translations = builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[0]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt_path is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1
                    gold_attn_matrices.append(batch_data['gold_attention'])

                n_best_preds = [" ".join(pred)
                                for pred in trans.pred_sents[:self.n_best]]
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    os.write(1, output.encode('utf-8'))

                # Debug attention.
                if attn_debug:
                    srcs = trans.src_raw
                    preds = trans.pred_sents[0]
                    preds.append('</s>')
                    attns = trans.attns[0].tolist()
                    header_format = "{:>10.10} " + "{:>10.7} " * len(srcs)
                    row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    output = header_format.format("", *trans.src_raw) + '\n'
                    for word, row in zip(preds, attns):
                        max_index = row.index(max(row))
                        row_format = row_format.replace(
                            "{:>10.7f} ", "{:*>10.7f} ", max_index + 1)
                        row_format = row_format.replace(
                            "{:*>10.7f} ", "{:>10.7f} ", max_index)
                        output += row_format.format(word, *row) + '\n'
                        row_format = "{:>10.10} " + "{:>10.7f} " * len(srcs)
                    os.write(1, output.encode('utf-8'))

            # ADDED --------------------------------------------------------------
            duration = time.time()-start_time
            tot_time += duration
            tot_time_print = str(time.strftime("%H:%M:%S", time.gmtime(tot_time)))
            print("Batch {} - Duration: {:.2f} - Total: {}".format(ix,
                                                        duration,
                                                        tot_time_print))

        print("Final vocab size: ", len(self.fields["tgt"].vocab.itos))
            # END ----------------------------------------------------------------

        if self.report_score:
            self._report_score('PRED', pred_score_total, pred_words_total)
            if tgt_path is not None:
                self._report_score('GOLD', gold_score_total, gold_words_total)
                if self.report_bleu:
                    self._report_bleu(tgt_path)
                if self.report_rouge:
                    self._report_rouge(tgt_path)

        if self.dump_beam:
            import json
            json.dump(self.translator.beam_accum,
                      codecs.open(self.dump_beam, 'w', 'utf-8'))

        if self.dump_attn:
            attn_matrices = [a[0][0].cpu().numpy() for a in attn_matrices]
            gold_attn_matrices = [a['std'][:,0,:].data.cpu().numpy()
                              for a in gold_attn_matrices]
            pickle.dump({'pred': attn_matrices, 'gold': gold_attn_matrices},
                   open('attn_matrices_' + self.model_opt.attn_transform + '.out',
                        'wb'))


        return all_scores

    def translate_batch(self, batch, data, translation_pieces, models, len_orig_vocab):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           translation_pieces (dict): dictionary with the translation pieces
           models (list): list with the [ft, weights, bias] models/matrices
           len_orig_vocab (int): int with the length of the original vocab

        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab

        # ADDED ----------------------------------------------------
        if self.use_guided:

            # TEST - make things work with gpu
            if self.cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # List that will have the necessary translation pieces
            t_pieces = [translation_pieces[ix] for ix in batch.indices]

            # "Translate" the list into dictionaries indexed by word index
            tp_uni = list()
            tp_multi = list()
            out_uni = torch.zeros(batch.batch_size,
                                  len(vocab),
                                  dtype=torch.float,
                                  device=device)

            # To extend the vocab with the translation pieces
            added = 0
            init_index = len(vocab.itos)
      
            for ix, list_ in enumerate(t_pieces):
                aux_dict_uni = defaultdict(lambda: 0)
                aux_dict_multi = defaultdict(lambda: 0)
                # Sort the list_ by length_ of tuple_[0] so that all 1-grams new
                # subwords are added before being considered for the multi dictionary
                list_.sort(key=lambda x: len(x[0]))
                for tuple_ in list_:
                    # Tuple_ is ([list of words], weight)
                    key = str(vocab.stoi[tuple_[0][0]])
                    # Len of the list one, because we only want to add the uni-grams
                    # and all the members of n-grams are present as uni-grams
                    if len(tuple_[0]) == 1:
                        # Putted this to ifs together
                        if self.extend_with_tp and key == '0':
                            # Since the new uni-gram is not present it is going to
                            # be added to the itos and the corresponding mapping
                            # is going to be added to the dictionary stoi
                            vocab.itos.append(tuple_[0][0])
                            vocab.stoi[tuple_[0][0]] = init_index + added
                            # A new column of ones is added
                            out_uni = torch.cat((out_uni,
                                                 torch.zeros(out_uni.shape[0],
                                                             1,
                                                             dtype=torch.float)),
                                                1)
                            # Since the key is zero we have to add the new
                            # proper index instead of the key
                            aux_dict_uni[init_index+added] = tuple_[1]
                            out_uni[ix][int(init_index+added)] = tuple_[1]
                            # In the end, increase added. Just a way of printing this
                            # in the future, and not thinking about indexing
                            added += 1
                        
                        # If the word is already in the dictionary just add it 
                        else:
                            aux_dict_uni[key] = tuple_[1]
                            out_uni[ix][int(key)] = tuple_[1]
                    # If it is a n-gram, keep adding the new indeces to the key
                    # and adding them to the aux_dict_multi
                    else:
                        for word in tuple_[0][1:]:
                            if str(vocab.stoi[word])=='0': pdb.set_trace()
                            key += " " + str(vocab.stoi[word])
                        aux_dict_multi[key] = tuple_[1]
                tp_uni.append(aux_dict_uni)
                tp_multi.append(aux_dict_multi)
            
            # To repeat we have to make [1,2,3,1,2,3,...] because
            # in each beam we have batch_size translations, so we
            # want to add all the translation pieces once in each
            # beam. Therefore, we repeat them beam_size times and
            # then we flatten that list to get something of dimen
            # sion equal to beam_size * batch_size            
            #tp_uni_ = [tp_uni for _ in range(self.beam_size)]
            #tp_uni_rep = list(itertools.chain(*tp_uni_))
            #tp_multi_ = [tp_multi for _ in range(self.beam_size)]
            #tp_multi_rep = list(itertools.chain(*tp_multi_))
            out_uni_ = tuple([out_uni for _ in range(self.beam_size)])
            out_uni_rep = torch.cat(out_uni_, 0)

            if self.extend_with_tp and added:

                print("Added {} new words".format(added))
 
                # Add the extra necessary components
                unk_w = self.model.generator[0].weight[0]
                unk_b = self.model.generator[0].bias[0]

                if self.extend_pred:

                    id_w_mx = torch.empty(added, 
                                          self.model.generator[0].weight.shape[1],
                                          dtype=torch.float)

                    id_b_mx = torch.empty(added,
                                          dtype=torch.float)

                    if self.extend_pred == 'mlp': 
                     
                        def _predict(x, model_dict):
                            nr_layers = len(model_dict)//2
                            for i in range(1, nr_layers+1):
                                w = model_dict['fc'+str(i)+'.weight']
                                b = model_dict['fc'+str(i)+'.bias']
                                x = torch.matmul(w, x) + b
                            return x
        
                        for i in range(added):
                            try:
                                wv_ = torch.from_numpy(models[0].wv[self.fields['tgt'].vocab.itos[init_index+i]])
                                id_w_mx[i] = _predict(wv_, models[1])
                                id_b_mx[i] = _predict(wv_, models[2])
                            except:
                                id_w_mx[i] = unk_w
                                id_b_mx[i] = unk_b

                    elif self.extend_pred == 'ls':

                        for i in range(added):
                            try:
                                wv_ = torch.from_numpy(models[0].wv[self.fields['tgt'].vocab.itos[init_index+i]])
                                id_w_mx[i] = torch.matmul(wv_, models[1].float())
                                id_b_mx[i] = torch.matmul(wv_, models[2].float())
                            except:
                                id_w_mx[i] = unk_w
                                id_b_mx[i] = unk_b                                            

                else:
                    id_w_mx = unk_w.repeat([added, 1])
                    id_b_mx = unk_b.repeat([added])


                # Concat to the model
                model_weight = self.model.generator[0].weight
                model_bias = self.model.generator[0].bias

                self.model.generator[0].weight.data = torch.cat((model_weight,
                                                                 id_w_mx),
                                                                0)
                self.model.generator[0].bias.data = torch.cat((model_bias,
                                                               id_b_mx),
                                                              0)

                # debug - Using share decoder, updating one, updates the other
                #print("Generator: ", self.model.generator[0].weight.shape)
                #print("Decoder:   ", self.model.decoder.embeddings.emb_luts[0].weight.shape)

        def _ngrams(n_max, seq):
            """
            Args:
                n_max:
                seq:
        
            Outputs:
                final:
            """
            final = list()
            
            for n in range(2, n_max):
                for s in range(0, len(seq)-n+1):
                    if s+n > len(seq):
                        break
                    list_seq = [str(x.item()) for x in seq[s:s+n]]
                    final.append(list_seq)
            
            return final
        # END ------------------------------------------------------

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([vocab.stoi[t]
                                for t in self.ignore_when_blocking])

        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length,
                                    stepwise_penalty=self.stepwise_penalty,
                                    block_ngram_repeat=self.block_ngram_repeat,
                                    exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a): return Variable(a, volatile=True)

        def rvar(a): return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

            if 'fertility' in batch.__dict__:
                fertility = batch.fertility
            else:
                fertility = None
        else:
            fertility = None

        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(
            src, memory_bank, enc_states)

        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank.data)\
                                                  .long()\
                                                  .fill_(memory_bank.size(0))

        # (2) Repeat src objects `beam_size` times.
        src_map = rvar(batch.src_map.data) \
            if data_type == 'text' and self.copy_attn else None

        memory_bank = rvar(memory_bank.data)
        memory_lengths = src_lengths.repeat(beam_size)
        if fertility is not None:
            print("fertility (translator): ", fertility)
            fertility = var(fertility.data.repeat(1, beam_size))
        dec_states.repeat_beam_size_times(beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))

            # Turn any copied words to UNKs
            # 0 is unk
            if self.copy_attn:
                inp = inp.masked_fill(
                    inp.gt(len(self.fields["tgt"].vocab) - 1), 0)

            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            inp = inp.unsqueeze(2)

            # Run one step.
            dec_out, dec_states, attn = self.model.decoder(
                inp, memory_bank, dec_states, fertility=fertility, 
                memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)

            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch x beam word scores.
            if not self.copy_attn:
                out = self.model.generator.forward(dec_out).data

                # ADDED ----------------------------------------------------
                n_max = self.guided_n_max # n-gram max
                
                if self.use_guided:
                    # Deal with n-gram cases
                    bs = batch.batch_size
                    total_size = batch.batch_size * self.beam_size
                    out_multi = torch.zeros(total_size, len(vocab),
                                            device=device) 

                    if i > 0:
                        for j in range(len(beam)): 
                            if not len(tp_multi[j]): 
                                continue
                            for k, seq in enumerate(zip(*beam[j].next_ys)):
                                seq_ = [str(x.item()) for x in seq]
                                for l in range(1,n_max):
                                    if i - l < 1: 
                                        break
                                    if l == 1: 
                                        if " ".join(seq_[-l:]) not in tp_uni[j]: 
                                            break
                                    elif " ".join(seq_[-l:]) not in tp_multi[j]: 
                                        break
                                    for key, value in tp_multi[j].items():
                                        key_ = key.split()
                                        if len(key_)!=l+1: 
                                            continue
                                        if key_[-l-1:-1] == seq_[-l:]:
                                            w = int(key_[-1])
                                            if self.guided_correct_ngrams:
                                                if key_[-1] not in seq_:
                                                    out_multi[k*bs+j][w] += value
                                            else:
                                                out_multi[k*bs+j][w] += value

                        if self.guided_correct_1grams:
                            for j in range(len(beam)):
                                if not len(tp_uni[j]):
                                    continue
                                for k, seq in enumerate(zip(*beam[j].next_ys)):
                                    seq_ = [str(x.item()) for x in seq]
                                    for w in set(seq_):
                                        value = tp_uni[j][w]
                                        out_multi[k*bs+j][int(w)] -= value
                                    try:
                                        values = out_uni_rep[k*bs+j]+out_multi[k*bs+j]
                                        assert ((values >= 0.0) == True).all()
                                    except:
                                        print("ERROR: Problem when correcting 1-grams")
                                        pdb.set_trace()

                    # Modify the output layer
                    if self.extend_with_tp:
                        ldg1 = self.guided_1_weight
                        ldgn = self.guided_n_weight
                        lde1 = self.extend_1_weight
                        lden = self.extend_n_weight
                        out = torch.add(out,
                                        torch.cat((ldg1*out_uni_rep[:, :len_orig_vocab],
                                                   lde1*out_uni_rep[:, len_orig_vocab:]),
                                                  dim=1))
                        
                        out = torch.add(out,
                                        torch.cat((ldgn*out_multi[:, :len_orig_vocab],
                                                   lden*out_multi[:, len_orig_vocab:]),
                                                  dim=1))
                    
                    else:
                        out = torch.add(out, self.guided_1_weight*out_uni_rep)
                        out = torch.add(out, self.guided_n_weight*out_multi)
                # END ------------------------------------------------------

                out = unbottle(out)
                # beam x tgt_vocab
                beam_attn = unbottle(attn["std"])
            else:
                out = self.model.generator.forward(dec_out,
                                                   attn["copy"].squeeze(0),
                                                   src_map)
                # beam x (tgt_vocab + extra_vocab)
                out = data.collapse_copy_scores(
                    unbottle(out.data),
                    batch, self.fields["tgt"].vocab, data.src_vocabs)
                # beam x tgt_vocab
                out = out.log()
                beam_attn = unbottle(attn["copy"])
            # (c) Advance each beam.
            for j, b in enumerate(beam):
                b.advance(out[:, j],
                          beam_attn.data[:, j, :memory_lengths[j]])
                dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        ret["gold_score"] = [0] * batch_size
        if "tgt" in batch.__dict__:
            ret["gold_score"], ret["gold_attention"] = self._run_target(
                batch, data)
        ret["batch"] = batch

        if fertility is not None:
            cum_attn = ret['attention'][0][0].sum(0).squeeze(0).cpu().numpy()
            fert = fertility.data[:, 0].cpu().numpy()
            #for c, f in zip(cum_attn, fert):
            #    print('%f (%f)' % (c, f))
        #else:
            #print(ret['attention'][0][0].sum(0))

        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        if data_type == 'text':
            _, src_lengths = batch.src
        else:
            src_lengths = None
        src = onmt.io.make_features(batch, 'src', data_type)
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]

        #  (1) run the encoder on the src
        enc_states, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = \
            self.model.decoder.init_decoder_state(src, memory_bank, enc_states)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        tt = torch.cuda if self.cuda else torch
        gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
        dec_out, _, attn = self.model.decoder(
            tgt_in, memory_bank, dec_states, memory_lengths=src_lengths)

        tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
        for dec, tgt in zip(dec_out, batch.tgt[1:].data):
            # Log prob of each word.
            out = self.model.generator.forward(dec)
            tgt = tgt.unsqueeze(1)
            scores = out.data.gather(1, tgt)
            scores.masked_fill_(tgt.eq(tgt_pad), 0)
            gold_scores += scores

        return gold_scores, attn

    def _report_score(self, name, score_total, words_total):
        print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
            name, score_total / words_total,
            name, math.exp(-score_total / words_total)))

    def _report_bleu(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        print()

        res = subprocess.check_output("perl %s/tools/multi-bleu.perl %s"
                                      % (path, tgt_path, self.output),
                                      stdin=self.out_file,
                                      shell=True).decode("utf-8")

        print(">> " + res.strip())

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        res = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN"
            % (path, tgt_path),
            shell=True,
            stdin=self.out_file).decode("utf-8")
        print(res.strip())
