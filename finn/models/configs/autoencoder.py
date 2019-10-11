import torch.nn as nn


def _down_conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def _up_conv_block(in_channels, out_channels, kernel_size, stride, padding, output_padding):
    return nn.Sequential(
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding),
    )


def conv_autoencoder(input_shape, initial_hidden_channels, levels, encoded_dim):
    encoder = []
    decoder = []
    c_in, h, w = input_shape
    c_out = initial_hidden_channels

    for level in range(levels):
        if level != 0:
            c_in = c_out
            c_out *= 2

        encoder += _down_conv_block(c_in, c_out, kernel_size=3, stride=1, padding=1)
        encoder += _down_conv_block(c_out, c_out, kernel_size=4, stride=2, padding=1)

        decoder += _up_conv_block(c_out, c_in, kernel_size=3, stride=1, padding=1, output_padding=0)
        decoder += _up_conv_block(c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0)

        h //= 2
        w //= 2

    encoder.append(nn.Conv2d(c_out, encoded_dim, 1, 1, 0))
    decoder.append(nn.Conv2d(encoded_dim, c_out, 1, 1, 0))
    decoder = decoder[::-1]

    encoder = nn.Sequential(*encoder)
    decoder = nn.Sequential(*decoder)

    enc_shape = (encoded_dim, h, w)

    return encoder, decoder, enc_shape
