from torch import nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_planes)

        self.shortcut = None
        if in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)

        return out


class BottleneckConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, hidden_channels=512, residual=False):
        super().__init__()

        self.conv_first = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv_bottleneck = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv_final = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # Initialize final kernel to zero so the coupling layer initially performs
        # and identity mapping
        nn.init.zeros_(self.conv_final.weight)

        self.residual = residual
        if residual:
            if in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, stride=1, padding=0)
                nn.init.ones_(self.shortcut.weight)
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.conv_first(x))
        out = F.relu(self.conv_bottleneck(out))
        out = self.conv_final(out)

        if self.residual:
            residual = self.shortcut(x)
            out += residual

        return out
