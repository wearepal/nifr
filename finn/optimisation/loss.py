import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(features, lambda_=1.0):
    return GradReverse.apply(features, lambda_)


def contrastive_gradient_penalty(network, input, penalty_amount=1.0):
    """Contrastive gradient penalty.

    This is essentially the optimization introduced by Mescheder et al 2018.

    Args:
        network: Network to apply penalty through.
        input: Input or list of inputs for network.
        penalty_amount: Amount of penalty.

    Returns:
        torch.Tensor: gradient penalty optimization.

    """

    def _get_gradient(inp, output):
        gradient = torch.autograd.grad(
            outputs=output,
            inputs=inp,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        return gradient

    if not isinstance(input, (list, tuple)):
        input = [input]

    input = [inp.detach() for inp in input]
    input = [inp.requires_grad_() for inp in input]

    with torch.set_grad_enabled(True):
        output = network(*input)[-1]
    gradient = _get_gradient(input, output)
    gradient = gradient.view(gradient.size()[0], -1)
    penalty = (gradient ** 2).sum(1).mean()

    return penalty * penalty_amount


class PixelCrossEntropy(nn.CrossEntropyLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor):
        input = input.view(input.size(0), 256, *target.shape[1:])
        # make integer class labels
        target = (target * (256 - 1)).long()
        return super().forward(input, target)
