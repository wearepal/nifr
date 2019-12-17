from .bijector import Bijector


__all__ = ["Flatten", "ConstantAffine"]


class Flatten(Bijector):
    def __init__(self):
        super(Flatten, self).__init__()
        self.orig_shape = None

    def _forward(self, x, sum_ldj=None):
        self.orig_shape = x.shape

        y = x.flatten(start_dim=1)
        self.flat_shape = y.shape

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj

    def _inverse(self, y, sum_ldj=None):
        x = y.view(self.orig_shape)

        if sum_ldj is None:
            return x
        else:
            return x, sum_ldj


class ConstantAffine(Bijector):

    def __init__(self, scale, shift):
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def logdetjac(self):
        return self.scale.log().flatten().sum()

    def _forward(self, x, sum_ldj=None):
        y = self.scale * x + self.shift

        if sum_ldj is None:
            return y
        else:
            return y - self.logdetjac()

    def _inverse(self, y, sum_ldj=None):
        x = (y - self.shift) / self.scale

        if sum_ldj is None:
            return x
        else:
            return x + self.logdetjac()

