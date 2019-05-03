import torch.nn as nn

from layers.layer_utils import Flatten


class MnistConvClassifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        layers = []

        curr_channels = in_channels
        hidden_sizes = [20, 50]

        for hsize in hidden_sizes:
            layers.append(nn.Conv2d(curr_channels, hsize, 5, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            curr_channels = hsize

        layers.append(Flatten())

        layers.append(nn.Linear(4 * 4 * curr_channels, 500))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(500, 10))
        layers.append(nn.LogSoftmax(dim=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MnistConvNet(nn.Module):
    def __init__(self, in_channels, out_dims, hidden_sizes=(20, 50), kernel_size=3,
                 padding=1, output_activation=None):
        super().__init__()

        layers = []

        curr_channels = in_channels

        for hsize in hidden_sizes:
            layers.append(nn.Conv2d(curr_channels, hsize, kernel_size, stride=1, padding=padding))
            layers.append(nn.BatchNorm2d(hsize))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.MaxPool2d(2, 2))
            curr_channels = hsize

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(Flatten())

        layers.append(nn.Linear(curr_channels, curr_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.LayerNorm(curr_channels))
        layers.append(nn.Linear(curr_channels, out_dims))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
