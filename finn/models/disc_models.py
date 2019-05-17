import torch.nn as nn

from finn.layers import Flatten
from finn.models import MnistConvNet


def linear_classifier(in_channels, num_classes):
    return nn.Sequential(
        Flatten(),
        nn.Linear(in_channels, 512),
        nn.LayerNorm(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes),
        nn.LogSoftmax(dim=1))


def conv_classifier(in_channels, num_classes, depth):
    assert num_classes > 1
    output_activation = nn.LogSoftmax(dim=1)
    hidden_sizes = [in_channels * 16] * depth
    return MnistConvNet(in_channels, num_classes, hidden_sizes=hidden_sizes, kernel_size=3,
                        output_activation=output_activation)
