import cPickle as pickle
import numpy as np
import pdb
import sys

attention_file = sys.argv[1]
alignment_file = sys.argv[2]

d = pickle.load(open(attention_file, 'rb'))
attention = d['gold']

alignments = []
i = 0
f = open(alignment_file)
for line in f:
    fields = line.rstrip('\n').split()
    alignment = np.zeros_like(attention[i])

    #pdb.set_trace()
    for field in fields:
        pair = field.split('-')
        assert len(pair) == 2, pdb.set_trace()
        s = int(pair[0])
        t = int(pair[1])
        alignment[t, s] = 1.
    alignments.append(alignment)
    i += 1

assert len(attention) == len(alignments)
num_match = num_pred = num_match_hard = num_pred_hard = num_gold = 0.
for pred, gold in zip(attention, alignments):
    pred_hard = np.zeros_like(pred)
    pred_hard[range(pred.shape[0]), pred.argmax(1)] = 1.
    num_match_hard += (pred_hard*gold).sum()
    num_pred_hard += pred_hard.sum()
    num_match += (pred*gold).sum()
    num_pred += pred.sum()
    num_gold += gold.sum()

print 1. -  2*num_match / (num_pred + num_gold)
print 1. -  2*num_match_hard / (num_pred_hard + num_gold)
