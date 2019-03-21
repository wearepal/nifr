import torch.nn as nn
from torch.autograd import Function
from layers.mlp import Mlp


class GradReverse(Function):
    """Gradient reversal layer"""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.lambd), None


def _grad_reverse(features):
    return GradReverse.apply(features, 1.0)


class GradReverseDiscriminator(nn.Module):

    def __init__(self, mlp_sizes, activation=nn.ReLU, activation_out=nn.Sigmoid):
        super(GradReverseDiscriminator, self).__init__()
        self.mlp = Mlp(mlp_sizes, activation, activation_out)

    def forward(self, x):
        x = _grad_reverse(x)
        return self.mlp(x)
