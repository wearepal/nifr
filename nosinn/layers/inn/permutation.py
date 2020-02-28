from typing import Optional

import torch
from torch import Tensor

from nosinn.layers.inn import Bijector
from nosinn.utils import is_positive_int

__all__ = ["RandomPermutation", "ReversePermutation"]


class Permutation(Bijector):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.dim() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not is_positive_int(dim):
            raise ValueError("dim must be a positive integer.")

        super().__init__()
        self._dim = dim
        self.register_buffer("_permutation", permutation)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs, permutation, dim):
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError(
                "Dimension {} in inputs must be of size {}.".format(dim, len(permutation))
            )
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)

        return outputs

    def _forward(self, inputs, sum_ldj: Optional[Tensor] = None):
        y = self._permute(inputs, self._permutation, self._dim)

        return y, sum_ldj

    def _inverse(self, inputs, sum_ldj: Optional[Tensor] = None):
        y = self._permute(inputs, self._inverse_permutation, self._dim)

        return y, sum_ldj


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, in_channels, dim=1):
        if not is_positive_int(in_channels):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.randperm(in_channels), dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, in_channels, dim=1):
        if not is_positive_int(in_channels):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.arange(in_channels - 1, -1, -1), dim)
