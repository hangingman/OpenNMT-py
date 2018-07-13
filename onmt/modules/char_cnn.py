from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from .highway_layer import HighwayLayer


class CharEmbeddingsCNN(nn.Module):
    """A module to create word embeddings out of character
    embeddings, using convolutional layers.
    Arguments:
        embeddings(Embeddings) -- character embeddings
    """

    def __init__(self, opt, embeddings):
        super(CharEmbeddingsCNN, self).__init__()

        self.embeddings = embeddings

        self.char_embed_dim = opt.lm_char_vec_size
        self.filters = opt.lm_char_conv_filters

        self.convolutions = nn.ModuleList()

        n_filters = 0
        for i, (width, num) in enumerate(self.filters):
            conv = torch.nn.Conv1d(
                    in_channels=self.char_embed_dim,
                    out_channels=num,
                    kernel_size=width,
                    bias=True)
            n_filters += num

            self.convolutions.append(conv)

        if opt.lm_num_highway > 0:
            self.use_highway = True
            self.highway_layers = HighwayLayer(n_filters, opt.lm_num_highway)
        else:
            self.use_highway = False

        self.projection = nn.Linear(n_filters, opt.lm_word_vec_size)

    def forward(self, char_tgt):

        _, batch_size, max_chars, n_feats = char_tgt.size()
        # Reshaping the input to
        # (sequence_length * batch_size, max_chars_per_token, 1)
        emb = self.embeddings(char_tgt.view(-1,
                                            max_chars,
                                            n_feats))

        # (sequence_length * batch_size, embed_dim, max_chars_per_token)
        emb = emb.permute(0, 2, 1).contiguous()

        convs = []
        for conv in self.convolutions:
            convolved = conv(emb)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = F.relu(convolved)
            convs.append(convolved)

        # (sequence_length * batch_size, n_filters)
        word_embeddings = torch.cat(convs, dim=-1)

        if self.use_highway:
            # apply the highway layers
            # (sequence_length * batch_size, n_filters)
            word_embeddings = self.highway_layers(word_embeddings)

        # final projection  (sequence_length * batch_size, embedding_dim)
        word_embeddings = self.projection(word_embeddings)

        embedding_dim = word_embeddings.size(-1)

        # reshape to (sequence_length, batch_size, embedding_dim)
        word_embeddings = word_embeddings.view(-1, batch_size, embedding_dim)

        return word_embeddings
