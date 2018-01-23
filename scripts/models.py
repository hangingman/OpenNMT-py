from __future__ import division, print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pdb
import numpy as np
import utils

class BiLSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_fert, n_layers=2, dropOut=0.2, gpu=False):
        super(BiLSTMTagger, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_fert = max_fert
        self.n_layers = n_layers

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.n_layers, dropout=dropOut, bidirectional=True)
        
        # The linear layer that maps from hidden state space to tag space

        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.max_fert)

        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (utils.get_var(torch.zeros(self.n_layers * 2, 1, self.hidden_dim), gpu=True),
                utils.get_var(torch.zeros(self.n_layers * 2, 1, self.hidden_dim), gpu=True))
        
    def forward(self, sentence):
        wembs = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(wembs.view(len(sentence),1,-1) , self.hidden)
        fert_space = self.hidden2tag(lstm_out.view(wembs.size(0), -1))
        fert_scores = F.log_softmax(fert_space, dim=1)
        return fert_scores
