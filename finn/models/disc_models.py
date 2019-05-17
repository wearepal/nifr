import torch.nn as nn

from finn.layers import Flatten


def linear_classifier(in_channels, num_classes):
    return nn.Sequential(
        Flatten(),
        nn.Linear(in_channels, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.LayerNorm(1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, num_classes),
        nn.LogSoftmax(dim=1))
