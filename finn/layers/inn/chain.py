import torch
import torch.nn as nn

from finn.layers.inn.bijector import Bijector


class BijectorChain(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layer_list):
        super(BijectorChain, self).__init__()
        self.chain: nn.ModuleList = nn.ModuleList(layer_list)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx


class FactorOut(BijectorChain):
    """A generalized nn.Sequential container for normalizing flows
    with splitting.
    """

    @staticmethod
    def _compute_split_point(tensor, frac):
        return round(tensor.size(1) * frac)

    def _split_channelwise(self, tensor, frac):
        assert 0 <= frac <= 1
        split_point = self._compute_split_point(tensor, frac)
        return tensor.split(split_size=[tensor.size(1) - split_point, split_point], dim=1)

    def __init__(self, layer_list, splits=None):
        super().__init__(layer_list)
        self.splits: dict = splits or {}

    def _forward(self, x, logpx, inds=None):
        if inds is None:
            inds = range(len(self.chain))

        xs = []
        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=False)
                if i in self.splits:
                    x_removed, x = self._split_channelwise(x, self.splits[i])
                    xs.append(x_removed)
            xs.append(x)
            x = torch.cat(xs, dim=1)

            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=False)
                if i in self.splits:
                    x_removed, x = self._split_channelwise(x, self.splits[i])
                    xs.append(x_removed)
            xs.append(x)
            x = torch.cat(xs, dim=1)

            return x, logpx

    def _reverse(self, x, logpx=None, inds=None):
        len_chain = len(self.chain)
        if inds is None:
            inds = range(len_chain-1, -1, -1)

        components = {}
        for block_ind, frac in self.splits.items():
            x_removed, x = self._split_channelwise(x, frac=frac)
            components[block_ind] = x_removed

        if logpx is None:
            for i in inds:
                if i in components:
                    x = torch.cat([components[i], x], dim=1)
                x = self.chain[i](x, reverse=True)

            return x
        else:
            for i in inds:
                if i in components:
                    x = torch.cat([components[i], x], dim=1)
                x, logpx = self.chain[i](x, logpx, reverse=True)

            return x, logpx

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if reverse:
            return self._reverse(x, logpx=logpx, inds=inds)
        else:
            return self._forward(x, logpx=logpx, inds=inds)
