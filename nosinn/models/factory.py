from typing import Optional, List, Tuple

import numpy as np

from nosinn import layers
from nosinn.models import Classifier
from nosinn.configs import NosinnArgs


def build_fc_inn(
    args: NosinnArgs, input_shape: Tuple[int, ...], level_depth: Optional[int] = None
) -> layers.Bijector:
    """Build the model with ARGS.depth many layers

    If ARGS.glow is true, then each layer includes 1x1 convolutions.
    """
    input_dim = input_shape[0]
    level_depth = level_depth or args.level_depth

    chain: List[layers.Bijector] = [layers.Flatten()]
    for i in range(level_depth):
        if args.batch_norm:
            chain += [layers.MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag)]
        if args.glow:
            chain += [layers.InvertibleLinear(input_dim)]
        chain += [
            layers.MaskedCouplingLayer(
                input_dim=input_dim,
                hidden_dims=args.coupling_depth * [args.coupling_channels],
                mask_type="alternate",
                swap=(i % 2 == 0) and not args.glow,
                scaling=args.scaling,
            )
        ]

    # one last mixing of the channels
    if args.glow:
        chain += [layers.InvertibleLinear(input_dim)]
    else:
        chain += [layers.RandomPermutation(input_dim)]

    return layers.BijectorChain(chain)


def build_conv_inn(args: NosinnArgs, input_shape) -> layers.Bijector:

    input_dim = input_shape[0]

    def _block(_input_dim) -> layers.Bijector:
        _chain: List[layers.Bijector] = []
        if args.idf:
            _chain += [
                layers.IntegerDiscreteFlow(_input_dim, hidden_channels=args.coupling_channels)
            ]
            _chain += [layers.RandomPermutation(_input_dim)]
        else:
            if args.batch_norm:
                _chain += [layers.MovingBatchNorm2d(_input_dim, bn_lag=args.bn_lag)]
            if args.glow:
                _chain += [layers.Invertible1x1Conv(_input_dim, use_lr_decomp=True)]
            else:
                _chain += [layers.RandomPermutation(_input_dim)]

            if args.scaling == "none":
                _chain += [
                    layers.AdditiveCouplingLayer(
                        _input_dim,
                        hidden_channels=args.coupling_channels,
                        num_blocks=args.coupling_depth,
                        pcnt_to_transform=0.25,
                    )
                ]
            elif args.scaling == "sigmoid0.5":
                _chain += [
                    layers.AffineCouplingLayer(
                        _input_dim,
                        num_blocks=args.coupling_depth,
                        hidden_channels=args.coupling_channels,
                    )
                ]
            else:
                raise ValueError(f"Scaling {args.scaling} is not supported")

        return layers.BijectorChain(_chain)

    factor_splits: dict = {int(k): v for k, v in args.factor_splits.items()}
    factor_splits = {int(k): float(v) for k, v in factor_splits.items()}

    chain: List[layers.Bijector] = []
    if args.preliminary_level:
        chain.append(layers.BijectorChain([_block(input_dim) for _ in range(args.level_depth)]))
        if 0 in factor_splits:
            input_dim = round(factor_splits[0] * input_dim)

    for i in range(args.levels):
        level: List[layers.Bijector]
        if args.reshape_method == "haar":
            level = [layers.HaarDownsampling(input_dim)]
        else:
            level = [layers.SqueezeLayer(2)]
        input_dim *= 4

        level.extend([_block(input_dim) for _ in range(args.level_depth)])

        chain.append(layers.BijectorChain(level))
        j = i if not args.preliminary_level else i + 1
        if j in factor_splits:
            input_dim = round(factor_splits[j] * input_dim)

    chain: List[layers.Bijector] = [layers.FactorOut(chain, factor_splits)]

    # input_dim = int(np.product(input_shape))
    # chain += [layers.RandomPermutation(input_dim)]

    model = layers.BijectorChain(chain)

    return model


def build_discriminator(
    args, input_shape: Tuple[int, ...], frac_enc, model_fn, model_kwargs, optimizer_kwargs=None
):

    in_dim = input_shape[0]

    # this is done in models/inn.py
    if not args.train_on_recon and len(input_shape) > 2:
        in_dim = round(frac_enc * int(np.product(input_shape)))

    num_classes = args.y_dim if args.y_dim > 1 else 2
    discriminator = Classifier(
        model_fn(in_dim, args.y_dim, **model_kwargs),
        num_classes=num_classes,
        optimizer_kwargs=optimizer_kwargs,
    )

    return discriminator
