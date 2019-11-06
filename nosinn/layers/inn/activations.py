import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LogitTransform", "ZeroMeanTransform"]  # , "SigmoidTransform", "SoftplusTransform"]

_DEFAULT_ALPHA = 0  # 1e-6
_DEFAULT_BETA = 1.0


class ZeroMeanTransform(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            x = x + 0.5
            if logpx is None:
                return x
            return x, logpx
        else:
            x = x - 0.5
            if logpx is None:
                return x
            return x, logpx


class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha=_DEFAULT_ALPHA):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return _sigmoid(x, logpx, self.alpha)
        else:
            return _logit(x, logpx, self.alpha)


class SoftplusTransform(nn.Module):
    def __init__(self, beta=_DEFAULT_BETA):
        nn.Module.__init__(self)
        self.softplus = nn.Softplus(beta)

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            x -= 1e-8
            out = torch.log((x.exp() - 1) + 1.0e-8)
            # out = x
            log_det = ((1 / (1 - torch.exp(-x)) + 1.0e-8).log()).sum(1, keepdim=True)
            # log_det = 0
            return out, logpx + log_det
        else:
            out = self.softplus(x) + 1e-8
            # out = x
            log_det = F.logsigmoid(x).sum(1, keepdim=True)
            # log_det = 0
            return out, logpx + log_det


class SigmoidTransform(nn.Module):
    """Reverse of LogitTransform."""

    def __init__(self, start_dim=0, end_dim=None, alpha=_DEFAULT_ALPHA):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x, logpx=None, reverse=False):
        x_copy = x.clone()
        if reverse:
            x_copy[:, self.start_dim : self.end_dim], logpx = _logit(
                x[:, self.start_dim : self.end_dim], logpx, self.alpha
            )
        else:
            x_copy[:, self.start_dim : self.end_dim], logpx = _sigmoid(
                x[:, self.start_dim : self.end_dim], logpx, self.alpha
            )
        return x_copy, logpx


def _logit(x, logpx=None, alpha=_DEFAULT_ALPHA):
    s = alpha + (1 - 2 * alpha) * x
    y = torch.log(s) - torch.log(1 - s)
    if logpx is None:
        return y
    return y, logpx - _logdetgrad(x, alpha).view(x.size(0), -1).sum(1, keepdim=True)


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
