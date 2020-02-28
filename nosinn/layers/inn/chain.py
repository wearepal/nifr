from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .bijector import Bijector
from .misc import Flatten

__all__ = ["BijectorChain", "FactorOut", "OxbowNet"]


class BijectorChain(Bijector):
    """A generalized nn.Sequential container for normalizing flows."""

    def __init__(self, layer_list: List[Bijector]):
        super().__init__()
        self.chain: nn.ModuleList = nn.ModuleList(layer_list)
        self.reverse_chain: nn.ModuleList = nn.ModuleList(reversed(layer_list))

    def _forward(self, x, sum_ldj: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        for layer in self.chain:
            x, sum_ldj = layer(x, sum_ldj, reverse=False)
        return x, sum_ldj

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        for layer in self.reverse_chain:
            y, sum_ldj = layer(y, sum_ldj, reverse=True)
        return y, sum_ldj


class FactorOut(BijectorChain):
    """A generalized nn.Sequential container for normalizing flows with splitting."""

    splits: Dict[int, float]

    def __init__(self, layer_list, splits: Optional[Dict[int, float]] = None):
        # interleave Flatten layers into the layer list
        splits = splits or {}
        layer_list_interleaved = []
        # we have to compute new indexes because we have added additional elements
        # `adjusted_splits` contains the same information as `splits` but with the new indexes
        adjusted_splits = {}
        for i, layer in enumerate(layer_list):
            layer_list_interleaved.append(layer)
            if i in splits:
                layer_list_interleaved.append(Flatten())
                index_of_flatten = len(layer_list_interleaved) - 1  # get index of last element
                adjusted_splits[index_of_flatten] = splits[i]

        super().__init__(layer_list_interleaved)
        self.splits = adjusted_splits
        self._final_flatten = Flatten()
        self._chain_len: int = len(layer_list_interleaved)

    def _forward(self, x, sum_ldj: Optional[Tensor] = None):
        xs = []
        for i, layer in enumerate(self.chain):
            if i in self.splits:  # layer is a flatten layer
                x_removed, x = _frac_split_channelwise(x, self.splits[i])
                x_removed_flat, _ = layer(x_removed)
                xs.append(x_removed_flat)
            else:
                x, sum_ldj = layer(x, sum_ldj=sum_ldj, reverse=False)
        xs.append(self._final_flatten(x)[0])
        x = torch.cat(xs, dim=1)

        return x, sum_ldj

    def _inverse(self, y, sum_ldj: Optional[Tensor] = None):
        x = y

        components: Dict[int, Tensor] = {}

        # we only want access to the flatten layers, but we have to iterate over all layers here
        # because JIT does not allow indexing the module list
        for i, layer in enumerate(self.chain):
            # repeat the code from _forward to recover the shapes
            if i in self.splits:
                x_removed_flat, x = _frac_split_channelwise(x, self.splits[i])
                components[i], _ = layer(x_removed_flat, reverse=True)

        x, _ = self._final_flatten(x, reverse=True)

        for reverse_i, layer in enumerate(self.reverse_chain):
            i = self._chain_len - (reverse_i + 1)  # we need the index for the non-reverse chain
            if i in components:  # layer is a flatten layer
                x = torch.cat([components[i], x], dim=1)
            else:
                x, sum_ldj = layer(x, sum_ldj=sum_ldj, reverse=True)

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


def _compute_split_point(tensor: Tensor, frac: float) -> int:
    return int(round(tensor.size(1) * frac))


def _frac_split_channelwise(tensor: Tensor, frac: float):
    assert 0 <= frac <= 1
    split_point = _compute_split_point(tensor, frac)
    return tensor.split([tensor.size(1) - split_point, split_point], dim=1)
