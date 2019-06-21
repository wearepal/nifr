import torch
import torch.nn as nn

from pyro.distributions.util import broadcast_shape

from finn.layers.coupling import InvertibleLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


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


class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """

    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super(ConcatModule, self).__init__()

    def forward(self, *input_args):
        # we have a single object
        if len(input_args) == 1:
            # regardless of type,
            # we don't care about single objects
            # we just index into the object
            input_args = input_args[0]

        # don't concat things that are just single objects
        if torch.is_tensor(input_args):
            return input_args
        else:
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)


class ListOutModule(nn.ModuleList):
    """
    a custom module for outputting a list of tensors from a list of nn modules
    """

    def __init__(self, modules):
        super(ListOutModule, self).__init__(modules)

    def forward(self, *args, **kwargs):
        # loop over modules in self, apply same args
        return [mm(*args, **kwargs) for mm in self]


def call_nn_op(op):
    """
    a helper function that adds appropriate parameters when calling
    an nn module representing an operation like Softmax

    :param op: the nn.Module operation to instantiate
    :return: instantiation of the op module with appropriate parameters
    """
    if op in [nn.Softmax, nn.LogSoftmax]:
        return op(dim=1)
    else:
        return op()
