from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg

from .bijector import Bijector

__all__ = ["Invertible1x1Conv", "InvertibleLinear"]


class Invertible1x1Conv(Bijector):
    """Invertible 1x1 convolution"""

    def __init__(self, num_channels, use_lr_decomp=False):
        super().__init__()

        self.num_channels = num_channels
        self.use_lr_decomp = use_lr_decomp

        self.reset_parameters()

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_shape = (self.num_channels, self.num_channels)
        if not self.use_lr_decomp:
            w_init = np.linalg.qr(np.random.randn(*w_shape))[0]
            w_init = torch.from_numpy(w_init.astype("float32"))
            w_init = w_init.unsqueeze(-1).unsqueeze(-1)
            self.weight = nn.Parameter(w_init)
        else:
            np_w = linalg.qr(np.random.randn(*w_shape))[0].astype("float32")
            np_p, np_l, np_u = linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)

            self.register_buffer("p", torch.as_tensor(np_p))
            self.l = nn.Parameter(torch.as_tensor(np_l))
            self.register_buffer("sign_s", torch.as_tensor(np_sign_s))
            self.log_s = nn.Parameter(torch.as_tensor(np_log_s))
            self.u = nn.Parameter(torch.as_tensor(np_u))

            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            self.register_buffer("l_mask", torch.as_tensor(l_mask))

    def get_w(self, reverse=False):

        if not self.use_lr_decomp:
            if reverse:
                w_inv = self.weight.squeeze().inverse()
                return w_inv.unsqueeze(-1).unsqueeze(-1)
            else:
                return self.weight
        else:
            l = self.l * self.l_mask + torch.eye(self.num_channels, device=self.l.device)
            u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())

            if reverse:
                u_inv = u.inverse()
                l_inv = l.inverse()
                p_inv = self.p.inverse()
                w_inv = u_inv @ (l_inv @ p_inv)
                return w_inv.unsqueeze(-1).unsqueeze(-1)
            else:
                w = self.p @ (l @ u)
                return w.unsqueeze(-1).unsqueeze(-1)

    def logdetjac(self, x):
        if not self.use_lr_decomp:
            dlogdet = self.weight.squeeze().det().abs().log() * x.size(-2) * x.size(-1)
        else:
            dlogdet = self.log_s.sum() * (x.size(2) * x.size(3))
        return dlogdet

    def _forward(self, x, sum_ldj: Optional[torch.Tensor] = None):

        w = self.get_w(reverse=False)
        output = F.conv2d(x, w)

        if sum_ldj is None:
            return output, None
        else:
            dlogdet = self.logdetjac(x)
            return output, sum_ldj - dlogdet

    def _inverse(self, y, sum_ldj: Optional[torch.Tensor] = None):

        weight_inv = self.get_w(reverse=True)

        output = F.conv2d(y, weight_inv)

        if sum_ldj is None:
            return output, None
        else:
            dlogdet = self.logdetjac(y)
            return output, sum_ldj + dlogdet


class InvertibleLinear(Bijector):
    def __init__(self, dim):
        super(InvertibleLinear, self).__init__()
        self.weight = nn.Parameter(torch.eye(dim), requires_grad=True)

    def logdetjac(self):
        return torch.log(torch.abs(torch.det(self.weight.double()))).float()

    def _forward(self, x, sum_ldj: Optional[torch.Tensor] = None):
        y = F.linear(x, self.weight)
        if sum_ldj is None:
            return y, None
        else:
            return y, sum_ldj - self.logdetjac().expand_as(sum_ldj)

    def _inverse(self, x, sum_ldj: Optional[torch.Tensor] = None):
        y = F.linear(x, self.weight.double().inverse().float())
        if sum_ldj is None:
            return y, None
        else:
            return y, sum_ldj + self.logdetjac().expand_as(sum_ldj)
