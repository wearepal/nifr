from typing import Union

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from typing_extensions import Protocol

from nifr.layers.resnet import ResidualNet

__all__ = [
    "linear_disciminator",
    "mp_28x28_net",
    "mp_32x32_net",
    "mp_64x64_net",
    "fc_net",
    "ModelFn",
]


class ModelFn(Protocol):
    def __call__(
        self, input_dim: int, target_dim: int, **model_kwargs: Union[float, str, bool]
    ) -> nn.Module:
        ...


class GlobalAvgPool(nn.Module):
    def __init__(self, keepdim=True):
        super().__init__()
        self.keepdim = keepdim

    def forward(self, x):
        return x.flatten(start_dim=1).mean(dim=1, keepdim=self.keepdim)


def linear_disciminator(in_dim, target_dim, hidden_channels=512, num_blocks=4, use_bn=False):

    act = F.relu if use_bn else F.selu
    layers = [
        nn.Flatten(),
        ResidualNet(
            in_features=in_dim,
            out_features=target_dim,
            hidden_features=hidden_channels,
            num_blocks=num_blocks,
            activation=act,
            dropout_probability=0.0,
            use_batch_norm=use_bn,
        ),
    ]
    return nn.Sequential(*layers)


def mp_64x64_net(input_dim, target_dim, use_bn=True):
    def conv_block(in_dim, out_dim, kernel_size, stride, padding):
        _block = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if use_bn:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 32, 5, 1, 0))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(32, 64, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(64, 128, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(128, 256, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(256, 512, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers += [nn.Flatten()]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


def resnet_50_ft(input_dim, target_dim, freeze=True, pretrained=True):
    net = resnet50(pretrained=pretrained)
    # net = resnet18(pretrained=pretrained)
    if freeze:
        for param in net.parameters():
            param.requires_grad = False

    # net.fc = nn.Linear(512, target_dim)
    net.fc = nn.Linear(2048, target_dim)

    return net


def mp_32x32_net(input_dim: int, target_dim: int, use_bn: bool = True):
    def conv_block(in_dim, out_dim, kernel_size, stride, padding):
        _block = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if use_bn:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 64, 5, 1, 0))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(64, 128, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(128, 256, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(256, 512, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers += [nn.Flatten()]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


def mp_64x64_net(input_dim, target_dim, use_bn=True):
    def conv_block(in_dim, out_dim, kernel_size, stride, padding):
        _block = []
        _block += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if use_bn:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 32, 5, 1, 0))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(32, 64, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(64, 128, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(128, 256, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(256, 512, 3, 1, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers += [nn.Flatten()]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


def mp_28x28_net(input_dim, target_dim, use_bn=True):
    def conv_block(in_dim, out_dim, kernel_size, stride):
        _block = []
        _block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1)]
        if use_bn:
            _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.LeakyReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 64, 3, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(64, 128, 3, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(128, 256, 3, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(256, 512, 3, 1))
    layers += [nn.MaxPool2d(2, 2)]

    layers += [nn.Flatten()]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


def strided_28x28_net(input_dim, target_dim):
    def conv_block(in_dim, out_dim, kernel_size, stride):
        _block = []
        _block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1)]
        _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.ReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 64, 3, 1))
    layers.extend(conv_block(64, 64, 4, 2))

    layers.extend(conv_block(64, 128, 3, 1))
    layers.extend(conv_block(128, 128, 4, 2))

    layers.extend(conv_block(128, 256, 3, 1))
    layers.extend(conv_block(256, 256, 4, 2))

    layers.extend(conv_block(256, 512, 3, 1))
    layers.extend(conv_block(512, 512, 4, 2))

    layers += [nn.Flatten()]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


def fc_net(input_shape, target_dim, hidden_dims=None):
    hidden_dims = hidden_dims or []

    def fc_block(in_dim, out_dim):
        _block = []
        _block += [nn.Linear(in_dim, out_dim)]
        _block += [nn.SELU()]
        return _block

    layers = [nn.Flatten()]
    input_dim = int(np.product(input_shape))

    for output_dim in hidden_dims:
        layers.extend(fc_block(input_dim, output_dim))
        input_dim = output_dim

    layers.append(nn.Linear(input_dim, target_dim))

    return nn.Sequential(*layers)
