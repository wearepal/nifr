import torch.nn as nn

from finn.layers import Flatten


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
