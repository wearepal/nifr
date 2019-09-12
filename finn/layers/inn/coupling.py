import torch
import torch.nn as nn

from finn.layers.inn.bijector import Bijector
from finn.layers.conv import BottleneckConvBlock
from finn.utils import RoundSTE, sum_except_batch


class AffineCouplingLayer(Bijector):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.d = in_channels - (in_channels // 2)
        self.net_s_t = BottleneckConvBlock(
            in_channels=self.d,
            hidden_channels=hidden_channels,
            out_channels=(in_channels - self.d) * 2,
        )

    def logdetjac(self, scale):
        return sum_except_batch(scale.log(), keepdim=True)

    def _get_scale_and_shift_params(self, x):
        s_t = self.net_s_t(x[:, :self.d])
        scale, shift = s_t.chunk(2, dim=1)
        scale = scale.sigmoid()
        return scale, shift

    def _forward(self, x, sum_ldj=None):
        scale, shift = self._get_scale_and_shift_params(x)
        y1 = scale * x[:, self.d:] + shift
        y = torch.cat([x[:, :self.d], y1], 1)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj - self.logdetjac(scale)

    def _inverse(self, x, sum_ldj=None):
        scale, shift = self._get_scale_and_shift_params(x)
        y1 = (x[:, self.d:] - shift) / scale
        y = torch.cat([x[:, :self.d], y1], 1)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj + self.logdetjac(scale)


class AdditiveCouplingLayer(Bijector):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.d = in_channels - (in_channels // 2)
        self.net_t = BottleneckConvBlock(
            in_channels=self.d,
            hidden_channels=hidden_channels,
            out_channels=(in_channels - self.d),
        )

    def logdetjac(self):
        return 0

    def get_shift_param(self, x):
        return self.net_t(x[:, :self.d])

    def _forward(self, x, sum_ldj=None):
        shift = self.get_shift_param(x)

        y1 = x[:, self.d:] + shift
        y = torch.cat([x[:, :self.d], y1], 1)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj - self.logdetjac()

    def _inverse(self, x, sum_ldj=None):
        shift = self.get_shift_param(x)

        y1 = x[:, self.d:] - shift
        y = torch.cat([x[:, :self.d], y1], 1)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj + self.logdetjac()


class IntegerDiscreteFlow(AdditiveCouplingLayer):
    def __init__(self, in_channels, hidden_channels):
        super().__init__(in_channels, hidden_channels)
        self.d = round(0.75 * in_channels)
        self.net_t = BottleneckConvBlock(
            in_channels=self.d,
            hidden_channels=hidden_channels,
            out_channels=(in_channels - self.d),
        )

    def logdetjac(self):
        return 0

    def _get_shift_param(self, x):
        shift = self.net_t(x[:, :self.d])
        # Round with straight-through-estimator
        return RoundSTE.apply(shift)

    def _forward(self, x, sum_ldj=None):
        shift = self._get_shift_param(x)
        y1 = x[:, self.d:] + shift
        y = torch.cat([x[:, :self.d], y1], 1)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj - self.logdetjac()

    def _inverse(self, x, sum_ldj=None):
        shift = self._get_shift_param(x)

        y1 = x[:, self.d:] - shift
        y = torch.cat([x[:, :self.d], y1], 1)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj + self.logdetjac()


class MaskedCouplingLayer(Bijector):
    """Used in the tabular experiments."""

    def __init__(self, d, hidden_dims, mask_type='alternate', swap=False):
        nn.Module.__init__(self)
        self.d = d
        self.register_buffer('mask', sample_mask(d, mask_type, swap).view(1, d))
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
    if mask_type == 'alternate':
        # Index-based masking in MAF paper.
        mask = torch.zeros(dim)
        mask[::2] = 1
        if swap:
            mask = 1 - mask
        return mask
    elif mask_type == 'channel':
        # Masking type used in Real NVP paper.
        mask = torch.zeros(dim)
        mask[: dim // 2] = 1
        if swap:
            mask = 1 - mask
        return mask
    else:
        raise ValueError('Unknown mask_type {}'.format(mask_type))


def build_net(input_dim, hidden_dims, activation="relu"):
    dims = (input_dim,) + tuple(hidden_dims) + (input_dim,)
    activation_modules = {"relu": nn.ReLU(inplace=True), "tanh": nn.Tanh()}

    chain = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        chain.append(nn.Linear(in_dim, out_dim))
        if i < len(hidden_dims):
            chain.append(activation_modules[activation])
    return nn.Sequential(*chain)
