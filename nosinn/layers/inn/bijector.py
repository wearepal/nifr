from abc import abstractmethod
from typing import Union, Tuple, overload, Optional

import torch.nn as nn
from torch import Tensor

__all__ = ["Bijector", "InvertBijector"]


class Bijector(nn.Module):
    """Base class of an invertible layer"""

    # @overload
    # def _forward(self, x: Tensor, sum_ldj: None = ...) -> Tensor:
    #     ...

    # @overload
    # def _forward(self, x: Tensor, sum_ldj: Tensor) -> Tuple[Tensor, Tensor]:
    #     ...

    @abstractmethod
    def _forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass"""

    # @overload
    # def _inverse(self, y: Tensor, sum_ldj: None = ...) -> Tensor:
    #     ...

    # @overload
    # def _inverse(self, y: Tensor, sum_ldj: Tensor) -> Tuple[Tensor, Tensor]:
    #     ...

    @abstractmethod
    def _inverse(
        self, y: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Reverse pass"""

    # @overload  # type: ignore[override]
    # def forward(self, x: Tensor, sum_ldj: None = ..., reverse: bool = ...) -> Tensor:
    #     ...

    # @overload
    # def forward(self, x: Tensor, sum_ldj: Tensor, reverse: bool = ...) -> Tuple[Tensor, Tensor]:
    #     ...

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
