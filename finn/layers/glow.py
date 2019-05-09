import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg


class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 convolution"""
    def __init__(self, num_channels, use_lr_decomp=True):
        super().__init__()

        self.num_channels = num_channels
        self.use_lr_decomp = use_lr_decomp

        self.reset_parameters()

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_shape = (self.num_channels, self.num_channels)
        if not self.use_lr_decomp:
            w_init = np.linalg.qr(np.random.randn(*w_shape))[0]
            w_init = torch.from_numpy(w_init.astype('float32'))
            w_init = w_init.unsqueeze(-1).unsqueeze(-1)
            self.weight = nn.Parameter(w_init)
        else:
            np_w = linalg.qr(np.random.randn(*w_shape))[
                0].astype('float32')
            np_p, np_l, np_u = linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            self.register_buffer('p', torch.as_tensor(np_p))
            self.l = nn.Parameter(torch.as_tensor(np_l))
            self.register_buffer('sign_s', torch.as_tensor(np_sign_s))
            self.log_s = nn.Parameter(torch.as_tensor(np_log_s))
            self.u = nn.Parameter(torch.as_tensor(np_u))

            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            self.register_buffer('l_mask', torch.as_tensor(l_mask))

    def get_w(self, reverse=False):

        if not self.use_lr_decomp:
            if reverse:
                w_inv = torch.inverse(self.weight.squeeze())
                return w_inv.unsqueeze(-1).unsqueeze(-1)
            else:
                return self.weight
        else:
            l = self.l * self.l_mask + torch.eye(self.num_channels)
            u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())
            w = self.p @ (l @ u)

            if reverse:
                u_inv = u.inverse()
                l_inv = l.inverse()
                p_inv = self.p.inverse()
                w_inv = u_inv @ (l_inv @ p_inv)
                return w_inv.unsqueeze(-1).unsqueeze(-1)
            else:
                return w.unsqueeze(-1).unsqueeze(-1)

    def dlogdet(self, x):
        if not self.use_lr_decomp:
            dlogdet = self.weight.squeeze().det().abs().log() * x.size(-2) * x.size(-1)
        else:
            dlogdet = self.log_s.sum() * (x.size(2) * x.size(3))
        return dlogdet

    def forward(self, x, logpx=None, reverse=False):

        if not reverse:
            return self._forward(x, logpx)
        else:
            return self._reverse(x, logpx)

    def _forward(self, x, logpx):

        w = self.get_w(reverse=False)
        output = F.conv2d(x, w)

        if logpx is None:
            return output
        else:
            dlogdet = self.dlogdet(x)
            logpx += dlogdet
            return output, logpx

    def _reverse(self, x, logpx):

        weight_inv = self.get_w(reverse=True)

        output = F.conv2d(x, weight_inv)

        if logpx is None:
            return output
        else:
            dlogdet = self.dlogdet(x)
            logpx -= dlogdet
            return output, logpx


class BruteForceLayer(nn.Module):

    def __init__(self, dim):
        super(BruteForceLayer, self).__init__()
        self.weight = nn.Parameter(torch.eye(dim))

    def forward(self, x, logpx=None, reverse=False):

        if not reverse:
            y = F.linear(x, self.weight)
            if logpx is None:
                return y
            else:
                return y, logpx - self._logdetgrad.expand_as(logpx)

        else:
            y = F.linear(x, self.weight.double().inverse().float())
            if logpx is None:
                return y
            else:
                return y, logpx + self._logdetgrad.expand_as(logpx)

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.weight.double()))).float()


# ActNorm Layer with data-dependant init
class ActNorm(nn.Module):
    def __init__(self, num_features, logscale_factor=1., scale=1.):
        super(ActNorm, self).__init__()

        self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1)))
        self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1)))

    def forward_(self, input, objective):
        input_shape = input.size()
        input = input.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True
            unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = input.size(0) * input.size(-1)
            b = -torch.sum(input, dim=(0, -1)) / sum_size
            vars = unsqueeze(torch.sum((input + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6)) / self.logscale_factor

            self.b.data.copy_(unsqueeze(b).data)
            self.logs.data.copy_(logs.data)

        logs = self.logs * self.logscale_factor
        b = self.b

        output = (input + b) * torch.exp(logs)
        dlogdet = torch.sum(logs) * input.size(-1)  # c x h

        return output.view(input_shape), objective - dlogdet


def _test():
    x = torch.randn(100, 3, 28, 28)
    conv = Invertible1x1Conv(3, use_lr_decomp=True)
    print(conv(x))
