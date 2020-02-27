from abc import abstractmethod
from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

__all__ = ["Bijector", "InvertBijector"]


class Bijector(nn.Module):
    """Base class of an invertible layer"""

    @abstractmethod
    def _forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass"""

    @abstractmethod
    def _inverse(
        self, y: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Reverse pass"""

    def forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None, reverse: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if reverse:
            return self._inverse(x, sum_ldj)
        else:
            return self._forward(x, sum_ldj)


class InvertBijector(Bijector):
    def __init__(self, to_invert: Bijector):
        super().__init__()
        self.to_invert = to_invert

    def forward(self, x: Tensor, sum_ldj: Optional[Tensor] = None, reverse: bool = False):
        return self.to_invert(x, sum_ldj=sum_ldj, reverse=not reverse)
