import torch.nn as nn


class _ResidualDownBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.downsample = None
        if inplanes != planes or stride != 1:
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.bn2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _ResidualUpBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                                        padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.upsample = None
        if inplanes != planes or stride != 1:
            self.upsample = nn.ConvTranspose2d(inplanes, planes, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.bn2(out)
        out = self.conv2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


def gated_conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.GLU(dim=1),
    )


def gated_up_conv(in_channels, out_channels, kernel_size, stride, padding, output_padding):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
        nn.GLU(dim=1),
    )


def conv_autoencoder(input_shape, initial_hidden_channels, levels, encoding_dim, decoding_dim, vae):
    encoder = []
    decoder = []
    c_in, h, w = input_shape
    c_out = initial_hidden_channels

    for level in range(levels):
        if level != 0:
            c_in = c_out
            c_out *= 2

        # encoder += [gated_conv(c_in, c_out, kernel_size=3, stride=1, padding=1)]
        # encoder += [gated_conv(c_out, c_out, kernel_size=4, stride=2, padding=1)]
        encoder += [_ResidualDownBlock(c_in, c_out, kernel_size=3)]
        encoder += [_ResidualDownBlock(c_out, c_out, stride=2, kernel_size=4)]

        # decoder += [gated_conv(c_out, c_in, kernel_size=3, stride=1, padding=1)]
        # decoder += [
        #     gated_up_conv(c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0)
        # ]
        decoder += [_ResidualDownBlock(c_out, c_in, kernel_size=3)]
        decoder += [_ResidualUpBlock(c_out, c_out, stride=2, kernel_size=4)]

        h //= 2
        w //= 2

    encoder_out_dim = 2 * encoding_dim if vae else encoding_dim

    encoder += [nn.Conv2d(c_out, encoder_out_dim, kernel_size=1, stride=1, padding=0)]
    decoder += [nn.Conv2d(encoding_dim, c_out, kernel_size=1, stride=1, padding=0)]
    decoder = decoder[::-1]
    decoder += [nn.Conv2d(input_shape[0], decoding_dim, kernel_size=1, stride=1, padding=0)]

    encoder = nn.Sequential(*encoder)
    decoder = nn.Sequential(*decoder)

    enc_shape = (encoding_dim, h, w)

    return encoder, decoder, enc_shape


def _linear_block(in_channels, out_channels):
    return nn.Sequential(nn.SELU(), nn.Linear(in_channels, out_channels))


def fc_autoencoder(input_shape, hidden_channels, levels, encoding_dim, vae):
    encoder = []
    decoder = []

    c_in = input_shape[0]
    c_out = hidden_channels

    for level in range(levels):
        encoder += [_linear_block(c_in, c_out)]
        decoder += [_linear_block(c_out, c_in)]
        c_in = c_out

    encoder_out_dim = 2 * encoding_dim if vae else encoding_dim

    encoder += [_linear_block(c_out, encoder_out_dim)]
    decoder += [_linear_block(encoding_dim, c_out)]
    decoder = decoder[::-1]

    encoder = nn.Sequential(*encoder)
    decoder = nn.Sequential(*decoder)

    enc_shape = (encoding_dim,)

    return encoder, decoder, enc_shape
