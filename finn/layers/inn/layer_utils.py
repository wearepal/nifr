import torch
import torch.nn as nn

from finn.layers.inn.inv_layer import InvertibleLayer


class InvFlatten(InvertibleLayer):
    def __init__(self):
        super(InvFlatten, self).__init__()
        self.orig_shape = None

    def _forward(self, x, sum_logdet=None, reverse=False):
        self.orig_shape = x.shape

        y = x.flatten(start_dim=1)

        if sum_logdet is None:
            return y
        else:
            return y, sum_logdet

    def _inverse(self, x, sum_logdet=None):
        y = x.view(self.orig_shape)

        if sum_logdet is None:
            return y
        else:
            return y, sum_logdet


class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, val):
        return torch.exp(val)
