from torch import nn as nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, out_planes, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, out_planes, eps=1e-4)

        self.shortcut = None
        if in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)

        return out


class BottleneckConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, out_channels=None):
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

    def forward(self, x):
        out = F.relu(self.conv_first(x))
        out = F.relu(self.conv_bottleneck(out))
        out = self.conv_final(out)
        return out
