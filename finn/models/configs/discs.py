import torch.nn as nn


def mp_28x28_classifier(input_dim, n_classes):

    def conv_block(in_dim, out_dim):
        _block = []
        _block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1)]
        _block += [nn.BatchNorm2d(out_dim)]
        _block += [nn.ReLU]
        return _block

    layers = []
    layers += [conv_block(input_dim, 64)]
    layers += [conv_block(input_dim, 64)]
    layers += [nn.MaxPool2d(2, 2)]

    layers += [conv_block(input_dim, 128)]
    layers += [conv_block(input_dim, 128)]
    layers += [nn.MaxPool2d(2, 2)]

    layers += [conv_block(input_dim, 256)]
    layers += [conv_block(input_dim, 256)]
    layers += [nn.MaxPool2d(2, 2)]

    layers += [conv_block(input_dim, 512)]
    layers += [conv_block(input_dim, 512)]
    layers += [nn.MaxPool2d(2, 2)]

    layers += [nn.Linear(512, n_classes)]

    return nn.Sequential(layers)
