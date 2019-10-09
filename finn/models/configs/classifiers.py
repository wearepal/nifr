import torch.nn as nn
import torch.nn.functional as F

from finn.layers.resnet import ResidualNet, ConvResidualNet


def linear_disciminator(in_dim, target_dim, hidden_channels=512, num_blocks=4, use_bn=False):
    def _conv_block(_in_channels, _out_channels, kernel_size=3, stride=1, padding=1):
        _block = [nn.Conv2d(_in_channels, _out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding)]
        if use_bn:
            _block.append(nn.BatchNorm2d(_out_channels))
        _block.append(nn.ReLU(inplace=True))

        return _block

    # layers = []
    # hidden_sizes = [in_dim * 16] * 2
    #
    # curr_dim = in_dim
    # for h in hidden_sizes:
    #     layers.extend(_conv_block(in_dim, h))
    #     curr_dim = h
    #
    # layers.append(nn.AdaptiveAvgPool2d(1))
    # layers.append(nn.Flatten())
    #
    # layers.append(nn.Linear(curr_dim, curr_dim))
    # layers.append(nn.ReLU(inplace=True))
    # layers.append(nn.Linear(curr_dim, target_dim))

    layers = [
        nn.Linear(in_dim, hidden_channels),
        nn.SELU(),
        nn.Linear(hidden_channels, target_dim)
    ]

    # layers = [
    #     ResidualNet(in_features=in_dim,
    #                 out_features=target_dim,
    #                 hidden_features=hidden_channels,
    #                 num_blocks=num_blocks,
    #                 activation=F.relu,
    #                 dropout_probability=0.,
    #                 use_batch_norm=use_bn),
    # ]

    # layers.extend(_conv_block(in_dim, 256, 3, 1))
    # layers.extend(_conv_block(256, 256, 4, 2))
    # layers.extend(_conv_block(256, 512, 3, 1))
    # layers.extend(_conv_block(512, 512, 4, 2))
    # #
    # # layers.append(nn.Conv2d(512, target_dim, kernel_size=1, stride=1, padding=0))
    # layers.append(nn.Flatten())
    # layers.append(nn.Linear(512, 512))
    # layers.append(nn.Linear(512, target_dim))

    return nn.Sequential(*layers)


def mp_32x32_net(input_dim, target_dim, use_bn=True):
    def conv_block(in_dim, out_dim, kernel_size, stride, padding):
        _block = []
        _block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                             stride=stride, padding=padding)]
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


def mp_28x28_net(input_dim, target_dim, use_bn=True):
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
