"""Test the invertible u-net"""
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torch import nn

from nifr.layers import OxbowNet


class Adder(nn.Module):
    def __init__(self, input_dim: int, number: float):
        super().__init__()
        self.number = torch.zeros(input_dim) + number

    def forward(self, x: float, sum_ldj: Optional[torch.Tensor] = None, reverse: bool = False):
        if reverse:
            return x - self.number, sum_ldj
        return x + self.number, sum_ldj


def test_unet():
    chain = [Adder(80, 1), Adder(16, 2), Adder(4, 4), Adder(2, 8)]
    splits = {0: 0.2, 1: 0.25, 2: 0.5}
    u_net = OxbowNet(chain, deepcopy(chain), splits=splits)
    input_ = torch.zeros(1, 80)
    output, _ = u_net(input_, reverse=False)
    reconstruction, _ = u_net(output, reverse=True)

    np.testing.assert_allclose(input_.numpy(), reconstruction.numpy())
