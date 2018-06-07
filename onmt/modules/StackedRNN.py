import torch
import torch.nn as nn
from torch.autograd import Variable


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[0][i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input, (h_1,)


class StackedGRUWithGalDropout(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRUWithGalDropout, self).__init__()
        self.gal_dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.hidden_size = rnn_size

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def sample_mask(self, batch_size, is_cuda):
        """Create a mask for recurrent dropout

        Arguments:
            batch_size(int) -- the size of the current batch
        """
        keep = 1.0 - self.gal_dropout
        self.mask = Variable(torch.bernoulli(
                        torch.Tensor(batch_size,
                                     self.hidden_size).fill_(keep)))

        if is_cuda:
            self.mask = self.mask.cuda()

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[0][i])
            input = h_1_i

            if self.gal_dropout > 0 and self.training:
                h_1_i = torch.mul(h_1_i, self.mask)
                h_1_i *= 1.0/(1.0 - self.gal_dropout)

            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input, (h_1,)
