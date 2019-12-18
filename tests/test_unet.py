"""Test the invertible u-net"""
import numpy as np

import torch
from torch import nn
from nosinn.layers import UNet


class Adder(nn.Module):
    def __init__(self, input_dim: int, number: float):
        super().__init__()
        self.number = torch.zeros(input_dim) + number

    def forward(self, x: float, sum_ldj=None, reverse: bool = False):
        if reverse:
            return x - self.number
        return x + self.number


def test_unet():
    chain = [Adder(16, 1), Adder(8, 2), Adder(4, 4), Adder(2, 8)]
    splits = {0: 0.5, 1: 0.5, 2: 0.5}
    u_net = UNet(chain, splits=splits)
    input_ = torch.zeros(1, 16)
    print(input_.size())
    output = u_net(input_, reverse=False)
    reconstruction = u_net(output, reverse=True)

    np.testing.assert_allclose(input_.numpy(), reconstruction.numpy())
