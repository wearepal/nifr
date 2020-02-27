from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from nosinn.utils import is_positive_int, sum_except_batch

from .misc import Bijector

__all__ = ["MovingBatchNorm1d", "MovingBatchNorm2d", "ActNorm"]


class MovingBatchNormNd(Bijector):
    def __init__(self, num_features, eps=1e-6, decay=0.1, bn_lag=0.0, affine=True):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer("step", torch.zeros(1))
        if self.affine:
            self.weight = Parameter(torch.zeros(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.zero_()
            self.bias.data.zero_()

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        c = x.size(1)
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()

        if self.training:
            # compute batch statistics
            x_t = x.transpose(0, 1).contiguous().view(c, -1)
            batch_mean = torch.mean(x_t, dim=1)
            batch_var = torch.var(x_t, dim=1)

            # moving average
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= 1.0 - self.bn_lag ** (self.step[0] + 1)
                used_var = batch_var - (1 - self.bn_lag) * (batch_var - used_var.detach())
                used_var /= 1.0 - self.bn_lag ** (self.step[0] + 1)

            # update running estimates
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.running_var -= self.decay * (self.running_var - batch_var.data)
            self.step += 1

        # perform normalization
        used_mean = used_mean.view(*self.shape).expand_as(x)
        used_var = used_var.view(*self.shape).expand_as(x)

        y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(x)
            bias = self.bias.view(*self.shape).expand_as(x)
            y = y * torch.exp(weight) + bias

        if sum_ldj is None:
            return y, None
        else:
            return y, sum_ldj - self.logdetgrad(x, used_var)

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        used_mean = self.running_mean
        used_var = self.running_var

        if self.affine:
            weight = self.weight.view(*self.shape).expand_as(y)
            bias = self.bias.view(*self.shape).expand_as(y)
            y = (y - bias) * torch.exp(-weight)

        used_mean = used_mean.view(*self.shape).expand_as(y)
        used_var = used_var.view(*self.shape).expand_as(y)
        x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean

        if sum_ldj is None:
            return x, None
        else:
            return x, sum_ldj + self.logdetgrad(x, used_var)

    def logdetgrad(self, x, used_var):
        ldj = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.weight.view(*self.shape).expand(*x.size())
            ldj += weight
        ldj = sum_except_batch(ldj, keepdim=True)
        return ldj

    def __repr__(self):
        return (
            "{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},"
            " affine={affine})".format(name=self.__class__.__name__, **self.__dict__)
        )


def stable_var(x, mean=None, dim=1):
    if mean is None:
        mean = x.mean(dim, keepdim=True)
    mean = mean.view(-1, 1)
    res = torch.pow(x - mean, 2)
    max_sqr = torch.max(res, dim, keepdim=True)[0]
    var = torch.mean(res / max_sqr, 1, keepdim=True) * max_sqr
    var = var.view(-1)
    # change nan to zero
    var[var != var] = 0
    return var


class MovingBatchNorm1d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1]


class MovingBatchNorm2d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1, 1, 1]


class ActNorm(Bijector):
    def __init__(self, features):
        """
        Transform that performs activation normalization. Works for 2D and 4D inputs. For 4D
        inputs (images) normalization is performed per-channel, assuming BxCxHxW input shape.
        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """
        if not is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        super().__init__()

        self.initialized = False
        self.log_scale = nn.Parameter(torch.zeros(features))
        self.shift = nn.Parameter(torch.zeros(features))

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def _broadcastable_scale_shift(self, inputs):
        if inputs.dim() == 4:
            return self.scale.view(1, -1, 1, 1), self.shift.view(1, -1, 1, 1)
        else:
            return self.scale.view(1, -1), self.shift.view(1, -1)

    def logdetjac(self, inputs):
        if inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            return h * w * torch.sum(self.log_scale)
        else:
            batch_size = inputs.size(0)
            return torch.sum(self.log_scale) * inputs.new_ones(batch_size)

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        if x.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        if self.training and not self.initialized:
            self._initialize(x)

        scale, shift = self._broadcastable_scale_shift(x)
        y = scale * x + shift

        if sum_ldj is not None:
            sum_ldj -= self.logdetjac(x)
        return y, sum_ldj

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        if y.dim() not in [2, 4]:
            raise ValueError("Expecting inputs to be a 2D or a 4D tensor.")

        scale, shift = self._broadcastable_scale_shift(y)
        x = (y - shift) / scale

        if sum_ldj is not None:
            sum_ldj += self.logdetjac(x)
        return y, sum_ldj

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)

        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
            self.log_scale.data = -torch.log(std)
            self.shift.data = -mu

        self.initialized = True
