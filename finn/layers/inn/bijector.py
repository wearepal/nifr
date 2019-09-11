from abc import abstractmethod
from typing import Union, Tuple

import torch.nn as nn
from torch import Tensor


class Bijector(nn.Module):
    """Base class of an invertible layer"""

    @abstractmethod
    def logdetjac(self, *args) -> Union[Tensor, float, int]:
        """Computation of the log determinant of the jacobian"""
    @abstractmethod
    def _forward(self, x, sum_logdet=None) -> Tuple[Tensor, Union[Tensor, Tensor]]:
        """Forward pass"""

    @abstractmethod
    def _inverse(self, x, sum_ldj=None) -> Tuple[Tensor, Union[Tensor, Tensor]]:
        """Reverse pass"""

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._inverse(x, logpx)
        else:
            return self._forward(x, logpx)
