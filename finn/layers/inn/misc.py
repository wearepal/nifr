from finn.layers.inn.bijector import Bijector


class Flatten(Bijector):
    def __init__(self):
        super(Flatten, self).__init__()
        self.orig_shape = None

    def logdetjac(self):
        return 0

    def _forward(self, x, sum_logdet=None, reverse=False):
        self.orig_shape = x.shape

        y = x.flatten(start_dim=1)

        if sum_logdet is None:
            return y
        else:
            return y, sum_logdet

    def _inverse(self, x, sum_ldj=None):
        y = x.view(self.orig_shape)

        if sum_ldj is None:
            return y
        else:
            return y, sum_ldj
