import math
from typing import Optional

import torch
from torch import Tensor

from .bijector import Bijector

# import torch.nn as nn
# import torch.nn.functional as F


__all__ = ["LogitTransform", "ZeroMeanTransform"]  # , "SigmoidTransform", "SoftplusTransform"]

_DEFAULT_ALPHA = 0  # 1e-6
_DEFAULT_BETA = 1.0


class ZeroMeanTransform(Bijector):
    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        x = x - 0.5
        return x, sum_ldj

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        x = y + 0.5
        return x, sum_ldj


class LogitTransform(Bijector):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha: float = _DEFAULT_ALPHA):
        super().__init__()
        self.alpha = alpha

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        return _logit(x, sum_ldj, self.alpha)

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        return _sigmoid(y, sum_ldj, self.alpha)


def _logit(x, sum_ldj: Optional[Tensor] = None, alpha: float = _DEFAULT_ALPHA):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    if sum_ldj is not None:
        sum_ldj -= _logdet_of_logit(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)
    return y, sum_ldj


def _sigmoid(y, sum_ldj: Optional[Tensor] = None, alpha: float = _DEFAULT_ALPHA):
    x = (torch.sigmoid(y) - alpha) / (1 - 2 * alpha)
    if sum_ldj is not None:
        sum_ldj += _logdet_of_logit(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)
    return x, sum_ldj


def _logdet_of_logit(x, alpha: float):
    s = alpha + (1 - 2 * alpha) * x
    logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * alpha)
    return logdetgrad
