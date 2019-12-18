from copy import deepcopy
from typing import List, Sequence, Optional

import torch
import torch.nn as nn
from .misc import Flatten
from .bijector import Bijector

__all__ = ["BijectorChain", "FactorOut", "UNet"]


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
                x_removed, x = _frac_split_channelwise(x, self.splits[i])
                x_removed_flat = self._factor_layers[i](x_removed)
                xs.append(x_removed_flat)
        xs.append(self._final_flatten(x))
        x = torch.cat(xs, dim=1)

        out = (x, sum_ldj) if sum_ldj is not None else x

        return out

    def _inverse(self, y, sum_ldj=None):
        inds = range(len(self.chain) - 1, -1, -1) if self.inds is None else self.inds
        x = y

        components = {}
        for block_ind, frac in self.splits.items():
            factor_layer = self._factor_layers[block_ind]
            split_point = factor_layer.flat_shape[1]
            x_removed_flat, x = x.split(split_size=[split_point, x.size(1) - split_point], dim=1)
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


class UNet(Bijector):
    """A generalized nn.Sequential container for normalizing flows with splitting in a U form."""

    def __init__(self, chain, splits=None, inds: Optional[Sequence] = None):
        super().__init__()
        self.down_chain = nn.ModuleList(chain)
        self.up_chain = nn.ModuleList(deepcopy(chain))
        self.splits: dict = splits or {}
        self.inds = inds

    def _forward(self, x: torch.Tensor, sum_ldj=None):
        down_inds = range(len(self.down_chain)) if self.inds is None else self.inds
        up_inds = range(len(self.up_chain) - 1, -1, -1) if self.inds is None else self.inds

        xs: List[torch.Tensor] = []
        # ============== contracting =========
        for i in down_inds:
            x = self.down_chain[i](x, sum_ldj=sum_ldj, reverse=False)
            if sum_ldj is not None:
                x, sum_ldj = x
            if i in self.splits:
                x_removed, x = _frac_split_channelwise(x, self.splits[i])
                xs.append(x_removed)
        xs.append(x)
        # import pdb; pdb.set_trace()
        # ============== expanding =========
        x = xs.pop()
        for i in up_inds:
            if i in self.splits:
                x_removed = xs.pop()
                x = torch.cat([x_removed, x], dim=1)
            x = self.up_chain[i](x, sum_ldj=sum_ldj, reverse=False)
            if sum_ldj is not None:
                x, sum_ldj = x

        out = (x, sum_ldj) if sum_ldj is not None else x

        return out

    def _inverse(self, y: torch.Tensor, sum_ldj=None):
        up_inds = range(len(self.up_chain)) if self.inds is None else self.inds
        down_inds = range(len(self.down_chain) - 1, -1, -1) if self.inds is None else self.inds

        ys: List[torch.Tensor] = []
        # ============== inverse expanding =========
        for i in up_inds:
            y = self.up_chain[i](y, sum_ldj=sum_ldj, reverse=False)
            if sum_ldj is not None:
                y, sum_ldj = y
            if i in self.splits:
                y_removed, y = _frac_split_channelwise(y, self.splits[i])
                ys.append(y_removed)
        ys.append(y)
        # ============== inverse contracting =========
        for i in down_inds:
            if i in self.splits:
                y_removed = ys.pop()
                y = torch.cat([y_removed, y], dim=1)
            y = self.down_chain[i](y, sum_ldj=sum_ldj, reverse=False)
            if sum_ldj is not None:
                y, sum_ldj = y

        out = (y, sum_ldj) if sum_ldj is not None else y

        return out


def _compute_split_point(tensor: torch.Tensor, frac: float):
    return round(tensor.size(1) * frac)


def _frac_split_channelwise(tensor: torch.Tensor, frac: float):
    assert 0 <= frac <= 1
    split_point = _compute_split_point(tensor, frac)
    print(f"split_point: {split_point}")
    return tensor.split(split_size=[tensor.size(1) - split_point, split_point], dim=1)
