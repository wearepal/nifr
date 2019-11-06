from abc import abstractmethod
from typing import Union, Tuple, overload, Optional

import torch.nn as nn
from torch import Tensor

__all__ = ["Bijector"]


class Bijector(nn.Module):
    """Base class of an invertible layer"""

    @overload
    def _forward(self, x: Tensor, sum_ldj: None = ...) -> Tensor:
        ...

    @overload
    def _forward(self, x: Tensor, sum_ldj: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def _forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Forward pass"""

    @overload
    def _inverse(self, y: Tensor, sum_ldj: None = ...) -> Tensor:
        ...

    @overload
    def _inverse(self, y: Tensor, sum_ldj: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def _inverse(
        self, y: Tensor, sum_ldj: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Reverse pass"""

    @overload  # type: ignore[override]
    def forward(self, x: Tensor, sum_ldj: None = ..., reverse: bool = ...) -> Tensor:
        ...

    @overload
    def forward(self, x: Tensor, sum_ldj: Tensor, reverse: bool = ...) -> Tuple[Tensor, Tensor]:
        ...

    def forward(
        self, x: Tensor, sum_ldj: Optional[Tensor] = None, reverse: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if reverse:
            return self._inverse(x, sum_ldj)
        return self._forward(x, sum_ldj)
