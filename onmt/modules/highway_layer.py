from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayLayer(nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated
    combination of a linear transformation and a non-linear transformation
    of its input.  :math:`y = g * x + (1 - g) * f(A(x))`, where :math:`A` is
    a linear transformation, :math:`f` is an element-wise non-linearity, and
    :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.
    This module will apply a fixed number of highway layers to its input,
    returning the final result.
    Heavily inspired on the implementation found in AllenNLP library
    (https://github.com/allenai/allennlp)
    Parameters
    ----------
    input_dim(int) -- The dimensionality of the input representation.
    num_layers(int) -- The number of highway layers to apply to the input.
    """
    def __init__(self, input_dim, num_layers=1):
        super(HighwayLayer, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim,
                                                            input_dim * 2)
                                            for _ in range(num_layers)])

        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, :self._input_dim]
            gate = projected_input[:, self._input_dim:2 * self._input_dim]
            nonlinear_part = F.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input
