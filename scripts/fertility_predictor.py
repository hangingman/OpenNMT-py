from __future__ import division, print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import generate_fertilities as gen
import torch.optim as optim
import argparse
import pdb
import numpy as np
import os
import utils, models

torch.manual_seed(1)


parser = argparse.ArgumentParser()
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--mlp_dim", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--model_name", type=str, default="fertility_model")
parser.add_argument("--train_source_path", type=str, default="/projects/tir1/users/cmalaviy/OpenNMT-py/ja-en/bpe.kyoto-train.cln.low.sink.ja.preprocessed")
parser.add_argument("--dev_source_path", type=str, default="/projects/tir1/users/cmalaviy/OpenNMT-py/ja-en/bpe.kyoto-dev.low.sink.ja.preprocessed")
parser.add_argument("--test_source_path", type=str, default="/projects/tir1/users/cmalaviy/OpenNMT-py/ja-en/bpe.kyoto-test.low.sink.ja")
parser.add_argument("--train_alignments_path", type=str, default="/projects/tir1/users/cmalaviy/OpenNMT-py/ja-en/ja-en-preprocessed.align")
parser.add_argument("--dev_alignments_path", type=str, default="/projects/tir1/users/cmalaviy/OpenNMT-py/ja-en/dev-ja-en-preprocessed.align")
parser.add_argument("--patience", type=int, default=3)
parser.add_argument("--max_fert", type=int, default=10)
parser.add_argument("--test", action='store_true')
parser.add_argument("--write_fertilities", type=str)
parser.add_argument("--gpu", action='store_true')
args = parser.parse_args()
print(args)

print("Reading training data...")
training_data = utils.read_file(args.train_source_path)
#gen.generate_actual_fertilities(args.train_source_path, args.train_alignments_path, args.train_source_path + ".fert.actual")
training_ferts = utils.read_file(args.train_source_path + ".fert.actual", fert=True)

# Set no. of classes as max(max_fert, highest fertility of any word in training set)

for ferts in training_ferts:
    for fert in ferts:
        if fert>args.max_fert:
            args.max_fert = fert

print("Maximum fertility set to %d" %args.max_fert)

dev_data = utils.read_file(args.dev_source_path)
#gen.generate_actual_fertilities(args.dev_source_path, args.dev_alignments_path, args.dev_source_path + ".fert.actual")
dev_ferts = utils.read_file(args.dev_source_path + ".fert.actual", fert=True)

if args.test:
    test_data = utils.read_file(args.test_source_path)

word_to_ix = {}
for sent in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)


def main():

    ############################################################################################


    if not os.path.isfile(args.model_name):
        fert_model = models.BiLSTMTagger(args.emb_dim, args.hidden_dim, len(word_to_ix), 
                                        args.max_fert, args.n_layers, args.dropout, args.gpu)
        if args.gpu:
            fert_model = fert_model.cuda()
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(fert_model.parameters(), lr=0.1)
        print("Training fertility predictor model...")
        patience_counter = 0
        prev_avg_tok_accuracy = 0

        for epoch in xrange(args.epochs):
            accuracies = []
            sent = 0
            tokens = 0
            cum_loss = 0
            print("Starting epoch %d .." %epoch)
            for sentence, ferts in zip(training_data, training_ferts):
                sent += 1
                tokens += len(sentence)
                if sent%100==0:
                    print("[Epoch %d] \
                        Sentence %d/%d, \
                        Tokens %d \
                        Cum_Loss: %f \
                        Average Accuracy: %f" 
                        % (epoch, sent, len(training_data), tokens,
                            cum_loss/tokens, sum(accuracies)/len(accuracies)))

                # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
                # before each instance
                fert_model.zero_grad()
                
                # Also, we need to clear out the hidden state of the LSTM, detaching it from its
                # history on the last instance.
                fert_model.hidden = fert_model.init_hidden()
            
                # Step 2. Get our inputs ready for the network, that is, turn them into Variables
                # of word indices.
                sentence_in = utils.prepare_sequence(sentence, word_to_ix, gpu=args.gpu)
                target_ferts = utils.prepare_sequence(ferts, gpu=args.gpu)


                # Step 3. Run our forward pass.
                fert_scores = fert_model(sentence_in)
                values, indices = torch.max(fert_scores, 1) 
                out_ferts = indices.cpu().data.numpy().flatten() + 1

                sent_acc = np.count_nonzero(out_ferts==target_ferts.cpu().data.numpy()) / out_ferts.shape[0]
                accuracies.append(sent_acc)

                # Step 4. Compute the loss, gradients, and update the parameters

                loss = loss_function(fert_scores, target_ferts-1)
                cum_loss += loss.cpu().data[0]
                loss.backward()
                optimizer.step()

            print("Loss: %f" % loss.cpu().data.numpy())
            print("Accuracy: %f" % np.mean(accuracies))
            print("Saving model..")
            torch.save(fert_model, args.model_name)
            print("Evaluating on dev set...")
            avg_tok_accuracy = eval(fert_model, epoch) 

            # Early Stopping
            if avg_tok_accuracy <= prev_avg_tok_accuracy:
                patience_counter += 1
                if patience_counter==args.patience:
                    print("Model hasn't improved on dev set for %d epochs. Stopping Training." % patience_counter)
                    break

            prev_avg_tok_accuracy = avg_tok_accuracy


    else:
        print("Loading tagger model from " + args.model_name + "...")
        fert_model = torch.load(args.model_name)
        if args.gpu:
            fert_model = fert_model.cuda()

    if args.test:
        out_path = args.write_fertilities if args.write_fertilities else args.test_source_path+".fert.predicted"
        test(fert_model, out_path)

def eval(fert_model, curEpoch=None):

    correct = 0
    toks = 0
    all_out_ferts = []
    # all_targets = np.array([])

    print("Starting evaluation on dev set... (%d sentences)" %  len(dev_data))
    for sentence, ferts in zip(dev_data, dev_ferts):
        fert_model.zero_grad()
        fert_model.hidden = fert_model.init_hidden()

        sentence_in = utils.prepare_sequence(sentence, word_to_ix, gpu=args.gpu)
        targets = utils.prepare_sequence(ferts, gpu=args.gpu)

        fert_scores = fert_model(sentence_in)
        values, indices = torch.max(fert_scores, 1)
        out_ferts = indices.cpu().data.numpy().flatten() + 1
        target_ferts = targets.cpu().data.numpy()
        correct += np.count_nonzero(out_ferts==target_ferts)
        toks += out_ferts.shape[0]
        all_out_ferts.append(out_ferts.tolist())
        # all_targets = np.append(all_targets, targets)

    avg_tok_accuracy = correct/toks
    print("Dev Set Accuracy: %f" %avg_tok_accuracy)

    return avg_tok_accuracy

        

def test(fert_model, out_path):

    toks = 0
    all_out_ferts = []

    print("Starting evaluation on test set... (%d sentences)" % len(test_data))
    for sentence in test_data:
        fert_model.zero_grad()
        fert_model.hidden = fert_model.init_hidden()

        sentence_in = utils.prepare_sequence(sentence, word_to_ix, gpu=args.gpu)

        fert_scores = fert_model(sentence_in)
        values, indices = torch.max(fert_scores, 1)
        out_ferts = indices.cpu().data.numpy().flatten() + 1
        toks += out_ferts.shape[0]
        all_out_ferts.append(out_ferts.tolist())
    
    print("Writing predicted fertility values..")
    # Write fertility values to file
    with open(out_path, 'w') as f:
        for ferts in all_out_ferts:
            for fert in ferts:
                f.write("%s " %fert)
            f.write("\n")


if __name__=="__main__":
    main()
