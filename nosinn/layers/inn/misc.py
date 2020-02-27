from typing import Optional
import torch
from torch import Tensor
from .bijector import Bijector


__all__ = ["Flatten", "ConstantAffine"]


class Flatten(Bijector):
    def __init__(self):
        super(Flatten, self).__init__()
        self.orig_shape = None

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        self.orig_shape = x.shape

        y = x.flatten(start_dim=1)
        self.flat_shape = y.shape

        if sum_ldj is None:
            return y, None
        else:
            return y, sum_ldj

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        x = y.view(self.orig_shape)

        if sum_ldj is None:
            return x, None
        else:
            return x, sum_ldj


class ConstantAffine(Bijector):
    def __init__(self, scale, shift):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale))
        self.register_buffer("shift", torch.as_tensor(shift))

    def logdetjac(self):
        return self.scale.log().flatten().sum()

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        y = self.scale * x + self.shift

        if sum_ldj is not None:
            sum_ldj -= self.logdetjac()
        return y, sum_ldj

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        x = (y - self.shift) / self.scale

        if sum_ldj is not None:
            sum_ldj += self.logdetjac()
        return x, sum_ldj
