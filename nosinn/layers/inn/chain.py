from typing import Dict, List, Sequence, Optional, Tuple, Union

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

    def _forward(self, x, sum_ldj=None):
        inds = range(len(self.chain)) if self.inds is None else self.inds
        return self.loop(inds=inds, reverse=False, x=x, sum_ldj=sum_ldj)

    def _inverse(self, y, sum_ldj=None):
        inds = range(len(self.chain) - 1, -1, -1) if self.inds is None else self.inds
        return self.loop(inds=inds, reverse=True, x=y, sum_ldj=sum_ldj)

    def reverse(self):         
        self.chain = self.chain[::-1]
        for child in self.chain:
            if hasattr(child, "reverse"):
                child.reverse()

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


class OxbowNet(Bijector):
    """A generalized nn.Sequential container for normalizing flows with splitting in a U form."""

    def __init__(
        self,
        down_chain: List[Bijector],
        up_chain: List[Bijector],
        splits: Dict[int, float],
        inds: Optional[Sequence] = None,
    ):
        super().__init__()
        self.down_chain = nn.ModuleList(down_chain)
        self.up_chain = nn.ModuleList(up_chain)
        self.splits: dict = splits or {}
        self.inds = inds

    def _forward(self, x: torch.Tensor, sum_ldj=None):

        # =================================== contracting =========================================
        result = self._contract(self.down_chain, x, sum_ldj=sum_ldj, reverse=False)
        if isinstance(result, tuple):
            xs, sum_ldj = result
        else:
            xs = result

        # ==================================== expanding ==========================================
        out = self._expand(self.up_chain, xs, sum_ldj=sum_ldj, reverse=False)

        return out

    def _inverse(self, y: torch.Tensor, sum_ldj=None):

        # ================================= inverse expanding =====================================
        result = self._contract(self.up_chain, y, sum_ldj=sum_ldj, reverse=True)
        if isinstance(result, tuple):
            ys, sum_ldj = result
        else:
            ys = result

        # ================================ inverse contracting ====================================
        out = self._expand(self.down_chain, ys, sum_ldj=sum_ldj, reverse=True)

        return out

    def _contract(
        self, chain, x, *, sum_ldj, reverse: bool
    ) -> Union[Tuple[List[Tensor], Tensor], List[Tensor]]:
        """Do a contracting loop

        Args:
            chain: a chain of INN blocks that expect smaller inputs as the chain goes on
            x: an input tensors
        """
        inds = range(len(chain)) if self.inds is None else self.inds

        xs: List[Tensor] = []
        for i in inds:
            x = chain[i](x, sum_ldj=sum_ldj, reverse=reverse)
            if sum_ldj is not None:
                x, sum_ldj = x
            if i in self.splits:
                x_removed, x = _frac_split_channelwise(x, self.splits[i])
                xs.append(x_removed)
        xs.append(x)  # save the last one as well
        return (xs, sum_ldj) if sum_ldj is not None else xs

    def _expand(self, chain, xs, *, sum_ldj, reverse: bool) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Do an expanding loop

        Args:
            chain: a chain of INN blocks that expect larger inputs as the chain goes on
            xs: a list with tensors where the sizes get larger as the list goes on
        """
        inds = range(len(chain) - 1, -1, -1) if self.inds is None else self.inds

        x = xs.pop()  # take the last in the list as the starting point
        for i in inds:
            if i in self.splits:  # if there was a split, we have to concatenate them together
                x_removed = xs.pop()
                x = torch.cat([x_removed, x], dim=1)
            x = chain[i](x, sum_ldj=sum_ldj, reverse=reverse)
            if sum_ldj is not None:
                x, sum_ldj = x

        return (x, sum_ldj) if sum_ldj is not None else x


def _compute_split_point(tensor: torch.Tensor, frac: float):
    return round(tensor.size(1) * frac)


def _frac_split_channelwise(tensor: torch.Tensor, frac: float):
    assert 0 <= frac <= 1
    split_point = _compute_split_point(tensor, frac)
    return tensor.split(split_size=[tensor.size(1) - split_point, split_point], dim=1)
