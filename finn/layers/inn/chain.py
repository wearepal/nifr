import torch
import torch.nn as nn
import numpy as np
from .misc import Flatten


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

    def _frac_split_channelwise(self, tensor, frac):
        assert 0 <= frac <= 1
        split_point = self._compute_split_point(tensor, frac)
        return tensor.split(
            split_size=[tensor.size(1) - split_point, split_point], dim=1
        )

    def __init__(self, layer_list, splits=None):
        super().__init__(layer_list)
        self.splits: dict = splits or {}
        self._factor_layers = {key: Flatten() for key in self.splits.keys()}
        self._final_flatten = Flatten()

    def _forward(self, x, logpx, inds=None):
        if inds is None:
            inds = range(len(self.chain))

        xs = []
        for i in inds:
            x = self.chain[i](x, logpx=logpx, reverse=False)
            if logpx is not None:
                x, logpx = x
            if i in self.splits:
                x_removed, x = self._frac_split_channelwise(x, self.splits[i])
                x_removed_flat = self._factor_layers[i](x_removed)
                xs.append(x_removed_flat)
        xs.append(self._final_flatten(x))
        x = torch.cat(xs, dim=1)

        out = (x, logpx) if logpx is not None else x

        return out

    def _reverse(self, x, logpx=None, inds=None):
        len_chain = len(self.chain)
        if inds is None:
            inds = range(len_chain - 1, -1, -1)

        components = {}
        for block_ind, frac in self.splits.items():
            factor_layer = self._factor_layers[block_ind]
            split_point = factor_layer.flat_shape[1]
            x_removed_flat, x = x.split(
                split_size=[split_point, x.size(1) - split_point], dim=1
            )
            components[block_ind] = factor_layer(x_removed_flat, reverse=True)

        x = self._final_flatten(x, reverse=True)

        for i in inds:
            if i in components:
                x = torch.cat([components[i], x], dim=1)
            x = self.chain[i](x, logpx=logpx, reverse=True)
            if logpx is not None:
                x, logpx = x

        out = (x, logpx) if logpx is not None else x

        return out

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if reverse:
            return self._reverse(x, logpx=logpx, inds=inds)
        else:
            return self._forward(x, logpx=logpx, inds=inds)
