import torch.nn as nn

from layers.layer_utils import Flatten


class MnistConvClassifier(nn.Module):

    def __init__(self, in_channels):
        super(MnistConvClassifier, self).__init__()

        hidden_sizes = [20, 50]
        layers = []

        curr_channels = in_channels

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
