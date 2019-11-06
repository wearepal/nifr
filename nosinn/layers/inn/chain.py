from typing import List, Sequence, Optional

import torch
import torch.nn as nn
from .misc import Flatten
from .bijector import Bijector

__all__ = ["BijectorChain", "FactorOut"]


class BijectorChain(Bijector):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layer_list: List[Bijector], inds: Optional[Sequence] = None):
        super().__init__()
        self.chain: nn.ModuleList = nn.ModuleList(layer_list)
        self.inds = inds

    def _forward(self, x, sum_ldj=None):
        inds = range(len(self.chain)) if self.inds is None else self.inds
        return self.loop(inds=inds, reverse=False, x=x, sum_ldj=sum_ldj)

    def _inverse(self, y, sum_ldj=None):
        inds = range(len(self.chain) - 1, -1, -1) if self.inds is None else self.inds
        return self.loop(inds=inds, reverse=True, x=y, sum_ldj=sum_ldj)

    def loop(self, inds, reverse: bool, x, sum_ldj=None):
        if sum_ldj is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, sum_ldj = self.chain[i](x, sum_ldj, reverse=reverse)
            return x, sum_ldj


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
        return tensor.split(split_size=[tensor.size(1) - split_point, split_point], dim=1)

    def __init__(self, layer_list, splits=None, inds: Optional[Sequence] = None):
        super().__init__(layer_list)
        self.splits: dict = splits or {}
        self._factor_layers = {key: Flatten() for key in self.splits.keys()}
        self._final_flatten = Flatten()
        self.inds = inds

    def _forward(self, x, sum_ldj=None):
        inds = range(len(self.chain)) if self.inds is None else self.inds

        xs = []
        for i in inds:
            x = self.chain[i](x, sum_ldj=sum_ldj, reverse=False)
            if sum_ldj is not None:
                x, sum_ldj = x
            if i in self.splits:
                x_removed, x = self._frac_split_channelwise(x, self.splits[i])
                x_removed_flat = self._factor_layers[i](x_removed)
                xs.append(x_removed_flat)
        xs.append(self._final_flatten(x))
        x = torch.cat(xs, dim=1)

        out = (x, sum_ldj) if sum_ldj is not None else x

        return out

    def _inverse(self, y, sum_ldj=None):
        inds = range(len(self.chain) - 1, -1, -1) if self.inds is None else self.inds

        components = {}
        for block_ind, frac in self.splits.items():
            factor_layer = self._factor_layers[block_ind]
            split_point = factor_layer.flat_shape[1]
            x_removed_flat, x = y.split(split_size=[split_point, y.size(1) - split_point], dim=1)
            components[block_ind] = factor_layer(x_removed_flat, reverse=True)

        x = self._final_flatten(x, reverse=True)

        for i in inds:
            if i in components:
                x = torch.cat([components[i], x], dim=1)
            x = self.chain[i](x, sum_ldj=sum_ldj, reverse=True)
            if sum_ldj is not None:
                x, sum_ldj = x

        out = (x, sum_ldj) if sum_ldj is not None else x

        return out
