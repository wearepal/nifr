import torch
import torch.nn as nn
import torch.nn.functional as F

from finn.layers.inn.inv_layer import InvertibleLayer
from finn.layers.resnet import ResidualBlock


class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, out_channels=None):
        super().__init__()

        self.conv_first = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv_bottleneck = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv_final = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # Initialize final kernel to zero so the coupling layer initially performs
        # and identity mapping
        nn.init.zeros_(self.conv_final.weight)

    def forward(self, x):
        out = F.relu(self.conv_first(x))
        out = F.relu(self.conv_bottleneck(out))
        out = self.conv_final(out)
        return out


class CouplingLayer(InvertibleLayer):
    def __init__(self, in_channels, hidden_channels, depth=1, swap=False):
        super().__init__()
        self.d = in_channels - (in_channels // 2)
        self.swap = swap
        self.net_s_t = ConvBlock(
            in_channels=self.d,
            hidden_channels=hidden_channels,
            out_channels=(in_channels - self.d) * 2,
        )
        # layers = []
        # in_planes = self.d
        # for _ in range(depth-1):
        #     layers += [ResidualBlock(in_planes, hidden_channels)]
        #     in_planes = hidden_channels
        # layers += [ResidualBlock(in_planes, (in_channels - self.d) * 2)]
        # self.net_s_t = nn.Sequential(*layers)

    def _forward(self, x, sum_logdet=None):
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.size(1) - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        y1 = (x[:, self.d:] * scale) + shift
        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else\
            torch.cat([y1, x[:, :self.d]], 1)

        if sum_logdet is None:
            return y
        else:
            delta_logp = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)
            return y, sum_logdet - delta_logp

    def _inverse(self, x, sum_logdet=None):
        in_dim = self.d
        out_dim = x.size(1) - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        y1 = (x[:, self.d:] - shift) / scale
        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else \
            torch.cat([y1, x[:, :self.d]], 1)

        if sum_logdet is None:
            return y
        else:
            delta_logp = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)
            return y, sum_logdet + delta_logp


class IntegerDiscreteFlow(InvertibleLayer):
    def __init__(self, in_channels, hidden_channels, depth=1, swap=False):
        super().__init__()
        self.d = in_channels - (in_channels // 2)
        self.swap = swap
        self.net_s_t = ConvBlock(
            in_channels=self.d,
            hidden_channels=hidden_channels,
            out_channels=(in_channels - self.d) * 2,
            n_layers=depth
        )

    def _forward(self, x, sum_logdet=None):
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.size(1) - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        y1 = (x[:, self.d:] * scale) + shift
        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else\
            torch.cat([y1, x[:, :self.d]], 1)

        if sum_logdet is None:
            return y
        else:
            logdet = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)
            return y, sum_logdet - logdet

    def _inverse(self, x, sum_logdet=None):
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.size(1) - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim] + 2.)
        shift = s_t[:, out_dim:]

        y1 = (x[:, self.d:] - shift) / scale
        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else \
            torch.cat([y1, x[:, :self.d]], 1)

        if sum_logdet is None:
            return y
        else:
            logdet = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)
            return y, sum_logdet + logdet


class MaskedCouplingLayer(InvertibleLayer):
    """Used in the tabular experiments."""

    def __init__(self, d, hidden_dims, mask_type='alternate', swap=False):
        nn.Module.__init__(self)
        self.d = d
        self.register_buffer('mask', sample_mask(d, mask_type, swap).view(1, d))
        self.net_scale = build_net(d, hidden_dims, activation="tanh")
        self.net_shift = build_net(d, hidden_dims, activation="relu")

    def _forward(self, x, sum_logdet=None):

        scale = torch.exp(self.net_scale(x * self.mask))
        shift = self.net_shift(x * self.mask)

        masked_scale = scale * (1 - self.mask) + torch.ones_like(scale) * self.mask
        masked_shift = shift * (1 - self.mask)

        logdetjac = torch.sum(torch.log(masked_scale).view(scale.shape[0], -1), 1, keepdim=True)

        y = x * masked_scale + masked_shift

        if sum_logdet is None:
            return y
        else:
            return y, sum_logdet - logdetjac

    def _inverse(self, x, sum_logdet=None):
        scale = torch.exp(self.net_scale(x * self.mask))
        shift = self.net_shift(x * self.mask)

        masked_scale = scale * (1 - self.mask) + torch.ones_like(scale) * self.mask
        masked_shift = shift * (1 - self.mask)

        logdetjac = torch.sum(torch.log(masked_scale).view(scale.shape[0], -1), 1, keepdim=True)

        y = (x - masked_shift) / masked_scale

        if sum_logdet is None:
            return y
        else:
            return y, sum_logdet + logdetjac


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
