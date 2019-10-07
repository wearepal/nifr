import torch.nn as nn


def mp_7x7_net(in_dim, target_dim):

    hidden_sizes = [in_dim * 16] * 3
    # return MnistConvNet(in_channels, num_classes, hidden_sizes=hidden_sizes, kernel_size=3,
    #                     output_activation=output_activation)

    layers = []

    curr_channels = in_dim

    for hsize in hidden_sizes:
        layers.append(nn.Conv2d(curr_channels, hsize, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(hsize))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.MaxPool2d(2, 2))
        curr_channels = hsize

    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())

    layers.append(nn.Linear(curr_channels, curr_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.LayerNorm(curr_channels))
    layers.append(nn.Linear(curr_channels, target_dim))

    # layers = []
    # layers.extend([
    #     nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=1),
    #     nn.BatchNorm2d(256),
    #     nn.LeakyReLU(inplace=True)
    # ])
    #
    # layers += [nn.MaxPool2d(2, 2)]
    #
    # layers.extend([
    #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    #     nn.BatchNorm2d(512),
    #     nn.LeakyReLU(inplace=True)
    # ])
    #
    # layers += [nn.MaxPool2d(2, 2)]
    #
    # layers.append(nn.Conv2d(512, target_dim, kernel_size=1, stride=1, padding=0))
    # layers.append(nn.Flatten())

    return nn.Sequential(*layers)


def mp_28x28_net(input_dim, target_dim):

    def conv_block(in_dim, out_dim):
        _block = []
        _block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)]
        _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.ReLU()]
        return _block

    layers = []
    layers.extend(conv_block(input_dim, 64))
    layers.extend(conv_block(64, 64))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(64, 128))
    layers.extend(conv_block(128, 128))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(128, 256))
    layers.extend(conv_block(256, 256))
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(conv_block(256, 512))
    layers.extend(conv_block(512, 512))
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
