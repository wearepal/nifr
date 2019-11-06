from abc import abstractmethod
from typing import Union, Tuple

import torch.nn as nn
from torch import Tensor

__all__ = ["Bijector"]


class Bijector(nn.Module):
    """Base class of an invertible layer"""

    @abstractmethod
    def logdetjac(self, *args) -> Union[Tensor, float, int]:
        """Computation of the log determinant of the jacobian"""

    @abstractmethod
    def _forward(self, x: Tensor, sum_ldj=None) -> Tuple[Tensor, Tensor]:
        """Forward pass"""

    @abstractmethod
    def _inverse(self, y: Tensor, sum_ldj=None) -> Tuple[Tensor, Tensor]:
        """Reverse pass"""

    def forward(self, x: Tensor, sum_ldj=None, reverse=False):
        if reverse:
            return self._inverse(x, sum_ldj)
        else:
            return self._forward(x, sum_ldj)
