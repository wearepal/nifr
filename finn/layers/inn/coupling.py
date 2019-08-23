import torch
import torch.nn as nn

from finn.layers.inn.inv_layer import InvertibleLayer


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, hidden_channels=512, out_channels=None, n_layers=3):
        super(ConvBlock, self).__init__()

        for i in range(n_layers):
            current_channels = in_channels if i == 0 else hidden_channels
            num_channels = out_channels if i == (n_layers - 1) else hidden_channels
            self.add_module(
                f'conv_{i}',
                nn.Conv2d(current_channels, num_channels, kernel_size=3, stride=1, padding=1),
            )
            self.add_module(f'bn{i}', nn.BatchNorm2d(num_channels))
            self.add_module(f'actfun_{i}', nn.ReLU(inplace=True))


class AffineCouplingLayer(InvertibleLayer):
    def __init__(self, in_channels, hidden_channels):
        super(AffineCouplingLayer, self).__init__()
        self.NN = ConvBlock(
            in_channels // 2, hidden_channels=hidden_channels[0], out_channels=in_channels
        )
        self.mask = None

    def _forward(self, x, logpx=None):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = (h[:, 1::2] + 2.0).sigmoid()
        z2 += shift
        z2 *= scale

        y = torch.cat([z1, z2], dim=1)
        if logpx is None:
            return y
        else:
            if self.mask is not None and x.requires_grad:
                x.register_hook(lambda grad: x * self.mask)
            delta_logp = scale.log().view(x.size(0), -1).sum(1, keepdim=True)
            return y, logpx - delta_logp

    def _reverse(self, x, logpx=None):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        shift = h[:, 0::2]
        scale = (h[:, 1::2] + 2.0).sigmoid()
        z2 /= scale
        z2 -= shift
        y = torch.cat([z1, z2], dim=1)

        if logpx is None:
            return y
        else:
            if self.mask is not None and x.requires_grad:
                x = x.register_hook(lambda grad: grad * self.mask)
            delta_logp = scale.log().view(x.size(0), -1).sum(1, keepdim=True)
            return y, logpx + delta_logp


class MaskedCouplingLayer(nn.Module):
    """Used in the tabular experiments."""

    def __init__(self, d, hidden_dims, mask_type='alternate', swap=False):
        nn.Module.__init__(self)
        self.d = d
        self.register_buffer('mask', sample_mask(d, mask_type, swap).view(1, d))
        self.net_scale = build_net(d, hidden_dims, activation="tanh")
        self.net_shift = build_net(d, hidden_dims, activation="relu")
        self.mask = None

    def forward(self, x, logpx=None, reverse=False):

        scale = torch.exp(self.net_scale(x * self.mask))
        shift = self.net_shift(x * self.mask)

        masked_scale = scale * (1 - self.mask) + torch.ones_like(scale) * self.mask
        masked_shift = shift * (1 - self.mask)

        if self.mask is not None and x.requires_grad:
            x.register_hook(lambda grad: grad * self.mask)

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
