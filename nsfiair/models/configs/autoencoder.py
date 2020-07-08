from typing import List, Tuple

import torch.nn as nn

__all__ = ["conv_autoencoder", "fc_autoencoder"]


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


def conv_autoencoder(
    input_shape,
    initial_hidden_channels: int,
    levels: int,
    encoding_dim,
    decoding_dim,
    vae,
    s_dim=0,
    level_depth: int = 2,
):
    assert level_depth in (2, 3), "only level depth 2 and 3 are supported right now"
    encoder: List[nn.Module] = []
    decoder: List[nn.Module] = []
    c_in, h, w = input_shape
    c_out = initial_hidden_channels

    for level in range(levels):
        if level != 0:
            c_in = c_out
            c_out *= 2

        encoder += [gated_conv(c_in, c_out, kernel_size=3, stride=1, padding=1)]
        if level_depth == 3:
            encoder += [gated_conv(c_out, c_out, kernel_size=3, stride=1, padding=1)]
        encoder += [gated_conv(c_out, c_out, kernel_size=4, stride=2, padding=1)]

        decoder += [gated_conv(c_out, c_in, kernel_size=3, stride=1, padding=1)]
        if level_depth == 3:
            decoder += [gated_conv(c_out, c_out, kernel_size=3, stride=1, padding=1)]
        decoder += [
            gated_up_conv(c_out, c_out, kernel_size=4, stride=2, padding=1, output_padding=0)
        ]

        h //= 2
        w //= 2

    encoder_out_dim = 2 * encoding_dim if vae else encoding_dim

    encoder += [nn.Conv2d(c_out, encoder_out_dim, kernel_size=1, stride=1, padding=0)]
    decoder += [nn.Conv2d(encoding_dim + s_dim, c_out, kernel_size=1, stride=1, padding=0)]
    decoder = decoder[::-1]
    decoder += [nn.Conv2d(input_shape[0], decoding_dim, kernel_size=1, stride=1, padding=0)]

    encoder = nn.Sequential(*encoder)
    decoder = nn.Sequential(*decoder)

    enc_shape = (encoding_dim, h, w)

    return encoder, decoder, enc_shape


def _linear_block(in_channels, out_channels):
    return nn.Sequential(nn.SELU(), nn.Linear(in_channels, out_channels))


def fc_autoencoder(
    input_shape: Tuple[int, ...],
    hidden_channels: int,
    levels: int,
    encoding_dim: int,
    vae: bool,
    s_dim: int = 0,
) -> Tuple[nn.Sequential, nn.Sequential, Tuple[int, ...]]:
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
    decoder += [_linear_block(encoding_dim + s_dim, c_out)]
    decoder = decoder[::-1]

    encoder = nn.Sequential(*encoder)
    decoder = nn.Sequential(*decoder)

    enc_shape = (encoding_dim,)

    return encoder, decoder, enc_shape
