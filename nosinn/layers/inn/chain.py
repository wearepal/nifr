from typing import Dict, List, Sequence, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from .misc import Flatten
from .bijector import Bijector

__all__ = ["BijectorChain", "FactorOut", "OxbowNet"]


class BijectorChain(Bijector):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layer_list: List[Bijector], inds: Optional[Sequence] = None):
        super().__init__()
        self.chain: nn.ModuleList = nn.ModuleList(layer_list)
        self.inds = inds

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        inds = range(len(self.chain)) if self.inds is None else self.inds
        return self.loop(inds=inds, reverse=False, x=x, sum_ldj=sum_ldj)

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        inds = range(len(self.chain) - 1, -1, -1) if self.inds is None else self.inds
        return self.loop(inds=inds, reverse=True, x=y, sum_ldj=sum_ldj)

    def loop(self, inds, reverse: bool, x, sum_ldj: Optional[Tensor] = None):
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

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        inds = range(len(self.chain)) if self.inds is None else self.inds

        xs = []
        for i in inds:
            x, sum_ldj = self.chain[i](x, sum_ldj=sum_ldj, reverse=False)
            if i in self.splits:
                x_removed, x = _frac_split_channelwise(x, self.splits[i])
                x_removed_flat, _ = self._factor_layers[i](x_removed)
                xs.append(x_removed_flat)
        xs.append(self._final_flatten(x)[0])
        x = torch.cat(xs, dim=1)

        return (x, sum_ldj)

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        inds = range(len(self.chain) - 1, -1, -1) if self.inds is None else self.inds
        x = y

        components = {}
        for block_ind, frac in self.splits.items():
            factor_layer = self._factor_layers[block_ind]
            split_point = factor_layer.flat_shape[1]
            x_removed_flat, x = x.split(split_size=[split_point, x.size(1) - split_point], dim=1)
            components[block_ind], _ = factor_layer(x_removed_flat, reverse=True)

        x, _ = self._final_flatten(x, reverse=True)

        for i in inds:
            if i in components:
                x = torch.cat([components[i], x], dim=1)
            x, sum_ldj = self.chain[i](x, sum_ldj=sum_ldj, reverse=True)

        return (x, sum_ldj)


class OxbowNet(Bijector):
    """A generalized nn.Sequential container for normalizing flows with splitting in a U form."""

    chain_len: int
    splits: Dict[int, float]

    def __init__(
        self, down_chain: List[Bijector], up_chain: List[Bijector], splits: Dict[int, float],
    ):
        super().__init__()
        assert len(down_chain) == len(up_chain)
        self.down_chain = nn.ModuleList(down_chain)
        self.up_chain = nn.ModuleList(up_chain)
        self.splits = splits or {}
        self.chain_len = len(self.down_chain)

    def _forward(self, x: Tensor, sum_ldj: Optional[Tensor] = None):

        # =================================== contracting =========================================
        xs, sum_ldj = self._contract(self.down_chain, x, sum_ldj=sum_ldj, reverse=False)

        # ==================================== expanding ==========================================
        out = self._expand(self.up_chain, xs, sum_ldj=sum_ldj, reverse=False)

        return out

    def _inverse(self, y: Tensor, sum_ldj: Optional[Tensor] = None):

        # ================================= inverse expanding =====================================
        ys, sum_ldj = self._contract(self.up_chain, y, sum_ldj=sum_ldj, reverse=True)

        # ================================ inverse contracting ====================================
        out = self._expand(self.down_chain, ys, sum_ldj=sum_ldj, reverse=True)

        return out

    def _contract(
        self, chain, x, *, sum_ldj, reverse: bool
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """Do a contracting loop

        Args:
            chain: a chain of INN blocks that expect smaller inputs as the chain goes on
            x: an input tensors
        """
        xs: List[Tensor] = []
        for i in range(self.chain_len):
            x, sum_ldj = chain[i](x, sum_ldj=sum_ldj, reverse=reverse)
            if i in self.splits:
                x_removed, x = _frac_split_channelwise(x, self.splits[i])
                xs.append(x_removed)
        xs.append(x)  # save the last one as well
        return (xs, sum_ldj)

    def _expand(self, chain, xs, *, sum_ldj, reverse: bool) -> Tuple[Tensor, Optional[Tensor]]:
        """Do an expanding loop

        Args:
            chain: a chain of INN blocks that expect smaller inputs as the chain goes on
            xs: a list with tensors where the sizes get smaller as the list goes on
        """
        x = xs.pop()  # take the last in the list as the starting point
        for i in range(self.chain_len - 1, -1, -1):
            if i in self.splits:  # if there was a split, we have to concatenate them together
                x_removed = xs.pop()
                x = torch.cat([x_removed, x], dim=1)
            x, sum_ldj = chain[i](x, sum_ldj=sum_ldj, reverse=reverse)

        return (x, sum_ldj)


def _compute_split_point(tensor: Tensor, frac: float):
    return round(tensor.size(1) * frac)


def _frac_split_channelwise(tensor: Tensor, frac: float):
    assert 0 <= frac <= 1
    split_point = _compute_split_point(tensor, frac)
    return tensor.split(split_size=[tensor.size(1) - split_point, split_point], dim=1)
