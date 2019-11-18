import math
import torch

# import torch.nn as nn
# import torch.nn.functional as F

from .bijector import Bijector

__all__ = ["LogitTransform", "ZeroMeanTransform"]  # , "SigmoidTransform", "SoftplusTransform"]

_DEFAULT_ALPHA = 0  # 1e-6
_DEFAULT_BETA = 1.0


class ZeroMeanTransform(Bijector):
    def _forward(self, x, sum_ldj=None):
        x = x - 0.5
        if sum_ldj is None:
            return x
        return x, sum_ldj

    def _inverse(self, y, sum_ldj=None):
        x = y + 0.5
        if sum_ldj is None:
            return x
        return x, sum_ldj


class LogitTransform(Bijector):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha=_DEFAULT_ALPHA):
        super().__init__()
        self.alpha = alpha

    def _forward(self, x, sum_ldj=None):
        return _logit(x, sum_ldj, self.alpha)

    def _inverse(self, y, sum_ldj=None):
        return _sigmoid(y, sum_ldj, self.alpha)


# class SoftplusTransform(nn.Module):
#     def __init__(self, beta=_DEFAULT_BETA):
#         super().__init__()
#         self.softplus = nn.Softplus(beta)

#     def forward(self, x, sum_ldj=None, reverse=False):
#         if reverse:
#             x -= 1e-8
#             out = torch.log((x.exp() - 1) + 1.0e-8)
#             # out = x
#             log_det = ((1 / (1 - torch.exp(-x)) + 1.0e-8).log()).sum(1, keepdim=True)
#             # log_det = 0
#             return out, sum_ldj + log_det
#         else:
#             out = self.softplus(x) + 1e-8
#             # out = x
#             log_det = F.logsigmoid(x).sum(1, keepdim=True)
#             # log_det = 0
#             return out, sum_ldj + log_det


# class SigmoidTransform(nn.Module):
#     """Reverse of LogitTransform."""

#     def __init__(self, start_dim=0, end_dim=None, alpha=_DEFAULT_ALPHA):
#         super().__init__()
#         self.alpha = alpha
#         self.start_dim = start_dim
#         self.end_dim = end_dim

#     def forward(self, x, sum_ldj=None, reverse=False):
#         x_copy = x.clone()
#         if reverse:
#             x_copy[:, self.start_dim : self.end_dim], sum_ldj = _logit(
#                 x[:, self.start_dim : self.end_dim], sum_ldj, self.alpha
#             )
#         else:
#             x_copy[:, self.start_dim : self.end_dim], sum_ldj = _sigmoid(
#                 x[:, self.start_dim : self.end_dim], sum_ldj, self.alpha
#             )
#         return x_copy, sum_ldj


def _logit(x, sum_ldj=None, alpha=_DEFAULT_ALPHA):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    if sum_ldj is None:
        return y
    return y, sum_ldj - _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)


def _sigmoid(y, logpy=None, alpha=_DEFAULT_ALPHA):
    y = y.clamp(-15, 9.9)
    x = (torch.sigmoid(y) - alpha) / (1 - 2 * alpha)
    if logpy is None:
        return x
    return x, logpy + _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)


def _logdetgrad(x, alpha):
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * alpha)
    return logdetgrad
