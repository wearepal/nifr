import torch
from torch.nn import functional as F


class RoundSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        return inputs.round()

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator
        """
        return grad_output


def to_discrete(inputs, dim=1):
    if inputs.dim() <= 1 or inputs.size(1) <= 1:
        return inputs.round()
    else:
        argmax = inputs.argmax(dim=1)
        return F.one_hot(argmax, num_classes=inputs.size(1))


def logit(p, eps=1e-8):
    p = p.clamp(min=eps, max=1.-eps)
    return torch.log(p / (1. - p))
