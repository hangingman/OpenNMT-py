from __future__ import division, print_function
import torch
from torch.autograd import Variable


def read_file(path, fert=False):
  
  """
   Reads source training file
   dev_or_train: read training data
  """
  with open(path) as f:
    lines = f.readlines()

  training_data = []
  for line in lines:
    if not fert:
      training_data.append(line.strip().split())
    else:
      training_data.append([int(fert) for fert in line.strip().split()])

  return training_data


def prepare_sequence(seq, to_ix=None, gpu=False):
    if isinstance(to_ix, dict):
      idxs = map(lambda w: to_ix[w], seq)
    elif isinstance(to_ix, list):
      # Temporary fix for unknown labels
      idxs = map(lambda w: to_ix.index(w) if w in to_ix else 0, seq)
    else:
      idxs = seq
    tensor = torch.LongTensor(idxs)
    return get_var(tensor, gpu)


def get_var(x,  gpu=False, volatile=False):
  x = Variable(x, volatile=volatile)
  return x.cuda() if gpu else x

