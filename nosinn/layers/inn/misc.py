from .bijector import Bijector

__all__ = ["Flatten"]


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
        y = y.view(self.orig_shape)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj
