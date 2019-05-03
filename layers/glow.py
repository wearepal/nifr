import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Invertible1x1Conv(nn.Conv2d):
    """Invertible 1x1 convolution"""
    def __init__(self, num_channels):
        self.num_channels = num_channels
        super().__init__(num_channels, num_channels, 1, bias=False)

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype('float32'))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward(self, x, logpx=None, reverse=False):
        if not reverse:
            return self._forward(x, logpx)
        else:
            return self._reverse(x, logpx)

    def _forward(self, x, logpx):
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        if logpx is None:
            return output
        else:
            dlogdet = self.weight.squeeze().det().abs().log() * x.size(-2) * x.size(-1)
            logpx -= dlogdet
            return output, logpx

    def _reverse(self, x, logpx):
        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        output = F.conv2d(x, weight_inv, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        if logpx is None:
            return output
        else:
            dlogdet = self.weight.squeeze().det().abs().log() * x.size(-2) * x.size(-1)
            logpx += dlogdet
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
