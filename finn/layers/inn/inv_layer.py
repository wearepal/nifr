from abc import abstractmethod

import torch.nn as nn


class InvertibleLayer(nn.Module):
    """Base class of an invertible layer"""

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._inverse(x, logpx)
        else:
            return self._forward(x, logpx)

    @abstractmethod
    def _forward(self, x, sum_logdet=None):
        """Forward pass"""

    @abstractmethod
    def _inverse(self, x, sum_logdet=None):
        """Reverse pass"""
