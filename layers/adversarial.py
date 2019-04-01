import torch.nn as nn
from torch.autograd import Function
from layers.mlp import Mlp


class GradReverse(Function):
    """Gradient reversal layer"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(features, lambda_):
    return GradReverse.apply(features, lambda_)


class GradReverseDiscriminator(nn.Module):

    def __init__(self, mlp_sizes, activation=nn.ReLU, activation_out=nn.Sigmoid):
        super(GradReverseDiscriminator, self).__init__()
        self.mlp = Mlp(mlp_sizes, activation, activation_out)

    def forward(self, x, lambda_=1.0):
        x = grad_reverse(x, lambda_)
        return self.mlp(x)
