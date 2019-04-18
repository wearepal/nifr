from abc import abstractmethod

import torch
import torch.nn as nn

from utils.utils import flatten_sum


class InvertibleLayer(nn.Module):
    """Base class of an invertible layer"""
    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._reverse(x, logpx)
        else:
            return self._forward(x, logpx)

    @abstractmethod
    def _forward(self, x, logpx):
        """Forward pass"""

    @abstractmethod
    def _reverse(self, x, logpx):
        """Reverse pass"""


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, hidden_channels=512, out_channels=None, n_layers=3):
        super(ConvBlock, self).__init__()

        for i in range(n_layers):
            current_channels = in_channels if i == 0 else hidden_channels
            num_channels = out_channels if i == (n_layers - 1) else hidden_channels
            self.add_module(f'conv_{i}', nn.Conv2d(current_channels, num_channels,kernel_size=3,
                                                   stride=1, padding=1))
            self.add_module(f'bn{i}', nn.BatchNorm2d(num_channels))
            self.add_module(f'actfun_{i}', nn.ReLU(inplace=True))


class AffineCouplingLayer(InvertibleLayer):
    def __init__(self, in_channels, hidden_channels):
        super(AffineCouplingLayer, self).__init__()
        self.NN = ConvBlock(in_channels // 2, hidden_channels=hidden_channels[0],
                            out_channels=in_channels)

    def _forward(self, x, logpx):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = (h[:, 1::2] + 2.).sigmoid()
        z2 += shift
        z2 *= scale

        delta_logp = scale.log().view(x.size(0), -1).sum(1, keepdim=True)

        return torch.cat([z1, z2], dim=1), logpx - delta_logp

    def _reverse(self, x, logpx):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = (h[:, 1::2] + 2.).sigmoid()
        z2 /= scale
        z2 -= shift

        delta_logp = scale.log().view(x.size(0), -1).sum(1, keepdim=True)

        return torch.cat([z1, z2], dim=1), logpx + delta_logp


class CouplingLayer(nn.Module):
    """Used in 2D experiments."""

    def __init__(self, d, intermediate_dim=64, swap=False):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )

    def forward(self, x, logpx=None, reverse=False):

        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = -logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = logdetjac

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


class MaskedCouplingLayer(nn.Module):
    """Used in the tabular experiments."""

    def __init__(self, d, hidden_dims, mask_type='alternate', swap=False):
        nn.Module.__init__(self)
        self.d = d
        self.register_buffer('mask', sample_mask(d, mask_type, swap).view(1, d))
        self.net_scale = build_net(d, hidden_dims, activation="tanh")
        self.net_shift = build_net(d, hidden_dims, activation="relu")

    def forward(self, x, logpx=None, reverse=False):

        scale = torch.exp(self.net_scale(x * self.mask))
        shift = self.net_shift(x * self.mask)

        masked_scale = scale * (1 - self.mask) + torch.ones_like(scale) * self.mask
        masked_shift = shift * (1 - self.mask)

        logdetjac = torch.sum(torch.log(masked_scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y = x * masked_scale + masked_shift
            delta_logp = -logdetjac
        else:
            y = (x - masked_shift) / masked_scale
            delta_logp = logdetjac

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


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
        mask[:dim // 2] = 1
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
