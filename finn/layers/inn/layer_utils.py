import torch
import torch.nn as nn

from finn.layers.inn.inv_layer import InvertibleLayer


class InvFlatten(InvertibleLayer):
    def __init__(self):
        super(InvFlatten, self).__init__()
        self.orig_shape = None

    def _forward(self, x, logpx=None, reverse=False):
        self.orig_shape = x.shape

        y = x.flatten(start_dim=1)

        if logpx is None:
            return y
        else:
            return y, logpx

    def _reverse(self, x, logpx=None):
        y = x.view(self.orig_shape)

        if logpx is None:
            return y
        else:
            return y, logpx


class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """

    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, val):
        return torch.exp(val)
