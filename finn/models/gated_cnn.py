import torch.nn as nn

from finn.layers import Flatten
from finn.layers.conv import GatedConv2d


class GatedConvClassifier(nn.Module):
    def __init__(self, in_channels, out_dims, hidden_sizes=None, kernel_size=3,
                 padding=1, activation=nn.Softplus(), output_activation=None):
        super().__init__()

        hidden_sizes = hidden_sizes or [512, 512, 512, 512]
        layers = []

        curr_channels = in_channels

        for hsize in hidden_sizes:
            layers.append(GatedConv2d(curr_channels, hsize, kernel_size=kernel_size, stride=1,
                                      padding=padding, activation=activation))
            curr_channels = hsize

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(Flatten())

        layers.append(nn.Linear(curr_channels, 1024))
        layers.append(activation)
        layers.append(nn.LayerNorm(1024))
        layers.append(nn.Linear(1024, out_dims))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
