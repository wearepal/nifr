import torch.nn as nn
import torch.nn.functional as F

from finn.layers.resnet import ResidualNet, ConvResidualNet


def mp_7x7_net(in_dim, target_dim):
    def _conv_block(_in_channels, _out_channels, kernel_size, stride):
        return [
            nn.Conv2d(_in_channels, _out_channels, kernel_size=kernel_size,
                      stride=stride, padding=1),
            # nn.BatchNorm2d(_out_channels),
            nn.LeakyReLU(inplace=True)
        ]

    layers = []
    # layers = [
    #     ConvResidualNet(
    #         in_channels=in_dim,
    #         out_channels=256,
    #         hidden_channels=256,
    #         num_blocks=4,
    #         activation=F.relu,
    #         dropout_probability=0,
    #         use_batch_norm=False),
    #     nn.AdaptiveAvgPool2d(1),
    #     nn.Flatten(),
    #     nn.Linear(256, 1024),
    #     nn.Linear(1024, target_dim)
    # ]
    layers.extend(_conv_block(in_dim, 256, 3, 1))
    layers.extend(_conv_block(256, 256, 4, 2))
    layers.extend(_conv_block(256, 512, 3, 1))
    layers.extend(_conv_block(512, 512, 4, 2))
    #
    layers.append(nn.Conv2d(512, target_dim, kernel_size=1, stride=1, padding=0))
    layers.append(nn.Flatten())
    # layers.append(nn.Linear(512, 512))
    # layers.append(nn.Linear(512, target_dim))

    return nn.Sequential(*layers)


def mp_28x28_net(input_dim, target_dim, use_bn =True):
    def conv_block(in_dim, out_dim, kernel_size, stride):
        _block = []
        _block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                             stride=stride, padding=1)]
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


def fc_net(input_dim, target_dim, hidden_dims=None):
    hidden_dims = hidden_dims or []

    def fc_block(in_dim, out_dim):
        _block = []
        _block += [nn.Linear(in_dim, out_dim)]
        _block += [nn.SELU()]
        return _block

    layers = [nn.Flatten()]

    for output_dim in hidden_dims:
        layers.extend(fc_block(input_dim, output_dim))
        input_dim = output_dim

    layers.append(nn.Linear(input_dim, target_dim))

    return nn.Sequential(*layers)
