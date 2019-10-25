from builtins import hasattr
from os.path import split, splitext
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nosinn.layers.conv import BottleneckConvBlock
from nosinn.layers.inn.bijector import Bijector
from nosinn.layers.resnet import ConvResidualNet, ResidualNet
from nosinn.utils import RoundSTE, sum_except_batch
from nosinn.utils.typechecks import is_probability


class CouplingLayer(Bijector):
    def __init__(self):
        super().__init__()
        self.d: int

    def logdetjac(self, *args):
        pass

    def _split(self, x):
        return x.split(split_size=(self.d, x.size(1) - self.d), dim=1)

    def _forward(self, x: Tensor, sum_ldj=None) -> Tuple[Tensor, Tensor]:
        pass

    def _inverse(self, y: Tensor, sum_ldj=None) -> Tuple[Tensor, Tensor]:
        pass


class AffineCouplingLayer(CouplingLayer):
    def __init__(self, in_channels, hidden_channels, num_blocks=2, pcnt_to_transform=0.5):
        assert is_probability(pcnt_to_transform)
        super().__init__()
        self.d = in_channels - round(pcnt_to_transform * in_channels)

        self.net_s_t = BottleneckConvBlock(
            in_channels=self.d,
            out_channels=(in_channels - self.d) * 2,
            hidden_channels=hidden_channels,
            use_bn=False,
        )

    def logdetjac(self, scale):
        return sum_except_batch(torch.log(scale), keepdim=True)

    def _scale_and_shift_fn(self, inputs):
        s_t = self.net_s_t(inputs)
        scale, shift = s_t.chunk(2, dim=1)
        scale = scale.sigmoid() + 0.5
        return scale, shift

    def _forward(self, x, sum_ldj=None):
        x_a, x_b = self._split(x)
        scale, shift = self._scale_and_shift_fn(x_a)
        y_b = scale * x_b + shift
        y = torch.cat([x_a, y_b], dim=1)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj - self.logdetjac(scale)

    def _inverse(self, y, sum_ldj=None):
        x_a, y_b = self._split(y)
        scale, shift = self._scale_and_shift_fn(x_a)
        x_b = (y_b - shift) / scale
        x = torch.cat([x_a, x_b], dim=1)

        if sum_ldj is None:
            return x
        else:
            return x, sum_ldj + self.logdetjac(scale)


class AdditiveCouplingLayer(CouplingLayer):
    def __init__(self, in_channels, hidden_channels, num_blocks=2, pcnt_to_transform=0.5):
        assert is_probability(pcnt_to_transform)

        super().__init__()
        self.d = in_channels - round(pcnt_to_transform * in_channels)

        self.net_t = BottleneckConvBlock(
            in_channels=self.d,
            out_channels=(in_channels - self.d),
            hidden_channels=hidden_channels,
            use_bn=False,
        )

    def logdetjac(self):
        return 0

    def _shift_fn(self, inputs):
        return self.net_t(inputs)

    def _forward(self, x, sum_ldj=None):
        x_a, x_b = self._split(x)
        shift = self._shift_fn(x_a)

        y_b = x_b + shift
        y = torch.cat([x_a, y_b], dim=1)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj - self.logdetjac()

    def _inverse(self, y, sum_ldj=None):
        x_a, y_b = self._split(y)
        shift = self._shift_fn(x_a)

        x_b = y_b - shift
        x = torch.cat([x_a, x_b], dim=1)

        if sum_ldj is None:
            return x
        else:
            return x, sum_ldj + self.logdetjac()


class IntegerDiscreteFlow(AdditiveCouplingLayer):
    def __init__(self, in_channels, hidden_channels, depth=3):
        super().__init__(in_channels, hidden_channels)
        self.d = round(0.75 * in_channels)

        self.net_t = ConvResidualNet(
            in_channels=self.d,
            out_channels=(in_channels - self.d),
            hidden_channels=hidden_channels,
            num_blocks=2,
            activation=F.relu,
            dropout_probability=0,
            use_batch_norm=False,
        )

    def _get_shift_param(self, inputs):
        shift = self.net_t(inputs)
        # Round with straight-through-estimator
        return RoundSTE.apply(shift)

    def _forward(self, x, sum_ldj=None):
        x_a, x_b = self._split(x)
        shift = self._get_shift_param(x_a)
        y_b = x_b + shift
        y = torch.cat([x_a, y_b], dim=1)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj - self.logdetjac()

    def _inverse(self, y, sum_ldj=None):
        x_a, y_b = self._split(y)
        shift = self._get_shift_param(x_a)

        x_b = y_b - shift
        x = torch.cat([x_a, x_b], dim=1)

        if sum_ldj is None:
            return x
        else:
            return x, sum_ldj + self.logdetjac()


class MaskedCouplingLayer(Bijector):
    """Used in the tabular experiments."""

    def __init__(self, d, hidden_dims, mask_type="alternate", swap=False):
        nn.Module.__init__(self)
        self.d = d
        self.register_buffer("mask", sample_mask(d, mask_type, swap).view(1, d))
        self.net_scale = build_net(d, hidden_dims, activation="tanh")
        self.net_shift = build_net(d, hidden_dims, activation="relu")

    def logdetjac(self, scale):
        return scale.log().view(scale.shape[0], -1).sum(dim=1, keepdim=True)

    def _forward(self, x, sum_ldj=None):

        scale = torch.exp(self.net_scale(x * self.mask))
        shift = self.net_shift(x * self.mask)

        masked_scale = scale * (1 - self.mask) + torch.ones_like(scale) * self.mask
        masked_shift = shift * (1 - self.mask)

        y = x * masked_scale + masked_shift

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj - self.logdetjac(masked_scale)

    def _inverse(self, x, sum_ldj=None):
        scale = torch.exp(self.net_scale(x * self.mask))
        shift = self.net_shift(x * self.mask)

        masked_scale = scale * (1 - self.mask) + torch.ones_like(scale) * self.mask
        masked_shift = shift * (1 - self.mask)

        y = (x - masked_shift) / masked_scale

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj + self.logdetjac(masked_scale)


def sample_mask(dim, mask_type, swap):
    if mask_type == "alternate":
        # Index-based masking in MAF paper.
        mask = torch.zeros(dim)
        mask[::2] = 1
        if swap:
            mask = 1 - mask
        return mask
    elif mask_type == "channel":
        # Masking type used in Real NVP paper.
        mask = torch.zeros(dim)
        mask[: dim // 2] = 1
        if swap:
            mask = 1 - mask
        return mask
    else:
        raise ValueError("Unknown mask_type {}".format(mask_type))


def build_net(input_dim, hidden_dims, activation="relu"):
    dims = (input_dim,) + tuple(hidden_dims) + (input_dim,)
    activation_modules = {"relu": nn.ReLU(inplace=True), "tanh": nn.Tanh()}

    chain = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        chain.append(nn.Linear(in_dim, out_dim))
        if i < len(hidden_dims):
            chain.append(activation_modules[activation])
    return nn.Sequential(*chain)
