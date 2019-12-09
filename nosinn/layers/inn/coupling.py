from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from nosinn.utils import RoundSTE, sum_except_batch
from nosinn.utils.typechecks import is_probability

from ..conv import BottleneckConvBlock
from ..resnet import ConvResidualNet
from .bijector import Bijector

__all__ = [
    "AdditiveCouplingLayer",
    "AffineCouplingLayer",
    "IntegerDiscreteFlow",
    "MaskedCouplingLayer",
]


class CouplingLayer(Bijector):
    d: int

    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def _split(self, x):
        return x.split([self.d, x.size(1) - self.d], dim=1)


class AffineCouplingLayer(CouplingLayer):
    def __init__(self, in_channels, hidden_channels, num_blocks=2, pcnt_to_transform=0.5):
        assert is_probability(pcnt_to_transform)
        super().__init__(d=in_channels - round(pcnt_to_transform * in_channels))

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
        shift = torch.clamp(shift, min=-5, max=5)
        scale = scale.sigmoid() + 0.5
        return scale, shift

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        x_a, x_b = self._split(x)
        scale, shift = self._scale_and_shift_fn(x_a)
        y_b = scale * x_b + shift
        y = torch.cat([x_a, y_b], dim=1)

        if sum_ldj is not None:
            sum_ldj -= self.logdetjac(scale)
        return y, sum_ldj

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        x_a, y_b = self._split(y)
        scale, shift = self._scale_and_shift_fn(x_a)
        x_b = (y_b - shift) / scale
        x = torch.cat([x_a, x_b], dim=1)

        if sum_ldj is not None:
            sum_ldj += self.logdetjac(scale)
        return x, sum_ldj


class AdditiveCouplingLayer(CouplingLayer):
    def __init__(self, in_channels, hidden_channels, num_blocks=2, pcnt_to_transform=0.5, d=None):
        assert is_probability(pcnt_to_transform)

        d = in_channels - round(pcnt_to_transform * in_channels) if d is None else d
        super().__init__(d=d)

        self.net_t = BottleneckConvBlock(
            in_channels=self.d,
            out_channels=(in_channels - self.d),
            hidden_channels=hidden_channels,
            use_bn=False,
        )

    def _shift_fn(self, inputs):
        return self.net_t(inputs)

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        x_a, x_b = self._split(x)
        shift = self._shift_fn(x_a)

        y_b = x_b + shift
        y = torch.cat([x_a, y_b], dim=1)

        return y, sum_ldj

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        x_a, y_b = self._split(y)
        shift = self._shift_fn(x_a)

        x_b = y_b - shift
        x = torch.cat([x_a, x_b], dim=1)

        return x, sum_ldj


class IntegerDiscreteFlow(AdditiveCouplingLayer):
    def __init__(self, in_channels, hidden_channels, depth=3):
        super().__init__(in_channels, hidden_channels, d=round(0.75 * in_channels))

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

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        x_a, x_b = self._split(x)
        shift = self._get_shift_param(x_a)
        y_b = x_b + shift
        y = torch.cat([x_a, y_b], dim=1)

        if sum_ldj is None:
            return y, None
        else:
            return y, sum_ldj - self.logdetjac()

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        x_a, y_b = self._split(y)
        shift = self._get_shift_param(x_a)

        x_b = y_b - shift
        x = torch.cat([x_a, x_b], dim=1)

        if sum_ldj is None:
            return x, None
        else:
            return x, sum_ldj + self.logdetjac()


class MaskedCouplingLayer(Bijector):
    """Used in the tabular experiments."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        mask_type: Literal["alternate", "channel"] = "alternate",
        swap: bool = False,
        scaling: Literal["none", "exp", "sigmoid0.5", "add2_sigmoid"] = "exp",
    ):
        super().__init__()
        # self.input_dim = input_dim
        self.register_buffer("mask", sample_mask(input_dim, mask_type, swap).view(1, input_dim))
        self.net_scale = build_net(input_dim, hidden_dims, activation="tanh")
        self.net_shift = build_net(input_dim, hidden_dims, activation="relu")
        self.scaling = scaling

    @staticmethod
    def logdetjac(scale):
        return scale.log().view(scale.shape[0], -1).sum(dim=1, keepdim=True)

    def _masked_scale_and_shift(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        shift = self.net_shift(x * self.mask)
        if self.scaling == "none":
            scale = torch.ones_like(shift)
        else:
            raw_scale = self.net_scale(x * self.mask)
            if self.scaling == "exp":
                scale = torch.exp(raw_scale)
            elif self.scaling == "sigmoid0.5":
                scale = torch.sigmoid(raw_scale) + 0.5
            elif self.scaling == "add2_sigmoid":
                scale = torch.sigmoid(raw_scale + 2.0)

        masked_scale = scale * (1 - self.mask) + torch.ones_like(scale) * self.mask
        masked_shift = shift * (1 - self.mask)
        return masked_scale, masked_shift

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        masked_scale, masked_shift = self._masked_scale_and_shift(x)

        y = x * masked_scale + masked_shift

        if sum_ldj is None:
            return y, None
        else:
            return y, sum_ldj - self.logdetjac(masked_scale)

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        masked_scale, masked_shift = self._masked_scale_and_shift(y)

        y = (y - masked_shift) / masked_scale

        if sum_ldj is None:
            return y, None
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


def build_net(input_dim: int, hidden_dims: Sequence[int], activation="relu"):
    dims = (input_dim,) + tuple(hidden_dims) + (input_dim,)
    activation_modules = {"relu": nn.ReLU(inplace=True), "tanh": nn.Tanh()}

    chain: List[nn.Module] = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        chain.append(nn.Linear(in_dim, out_dim))
        if i < len(hidden_dims):
            chain.append(activation_modules[activation])
    return nn.Sequential(*chain)
