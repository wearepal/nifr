import torch.nn as nn

from finn.layers import Flatten
from finn.models import MnistConvNet


def linear_classifier(in_channels, num_classes, hidden_channels=512,
                      output_activation=nn.LogSoftmax(dim=1)):
    return nn.Sequential(
        Flatten(),
        nn.Linear(in_channels, hidden_channels),
        nn.LayerNorm(hidden_channels),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_channels, hidden_channels),
        nn.LayerNorm(hidden_channels),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_channels, num_classes),
        output_activation)


def conv_classifier(in_channels, num_classes, depth):
    assert num_classes > 1
    output_activation = nn.LogSoftmax(dim=1)
    hidden_sizes = [in_channels * 16] * depth
    return MnistConvNet(in_channels, num_classes, hidden_sizes=hidden_sizes, kernel_size=3,
                        output_activation=output_activation)
