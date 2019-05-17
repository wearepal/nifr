import torch.nn as nn

from finn.layers import Flatten
from finn.layers.conv import GatedConv2d


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = None
        self.stride = stride

        if self.stride > 1:
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=1)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
