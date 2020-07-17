from typing import Dict, List, Optional, Tuple, Union

from torch import jit
from nifr import layers
from nifr.configs import InnArgs
from nifr.models import Classifier
from nifr.models.configs import ModelFn
from nifr.utils import product

__all__ = ["build_fc_inn", "build_conv_inn", "build_discriminator"]


def build_fc_inn(
    args: InnArgs, input_shape: Tuple[int, ...], level_depth: Optional[int] = None
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


def _block(args: InnArgs, input_dim: int) -> layers.Bijector:
    """Construct one block of the conv INN"""
    _chain: List[layers.Bijector] = []

    if args.idf:
        _chain += [layers.IntegerDiscreteFlow(input_dim, hidden_channels=args.coupling_channels)]
        _chain += [layers.RandomPermutation(input_dim)]
    else:
        if args.batch_norm:
            _chain += [layers.MovingBatchNorm2d(input_dim, bn_lag=args.bn_lag)]
        if args.glow:
            _chain += [layers.Invertible1x1Conv(input_dim, use_lr_decomp=True)]
        else:
            _chain += [layers.RandomPermutation(input_dim)]

        if args.scaling == "none":
            _chain += [
                layers.AdditiveCouplingLayer(
                    input_dim,
                    hidden_channels=args.coupling_channels,
                    num_blocks=args.coupling_depth,
                    pcnt_to_transform=0.25,
                )
            ]
        elif args.scaling == "sigmoid0.5":
            _chain += [
                layers.AffineCouplingLayer(
                    input_dim,
                    num_blocks=args.coupling_depth,
                    hidden_channels=args.coupling_channels,
                )
            ]
        else:
            raise ValueError(f"Scaling {args.scaling} is not supported")

    block = layers.BijectorChain(_chain)
    if args.jit:
        block = jit.script(block)
    return block


def _build_multi_scale_chain(
    args: InnArgs, input_dim, factor_splits, unsqueeze=False
) -> List[layers.Bijector]:
    chain: List[layers.Bijector] = []

    for i in range(args.levels):
        level: List[layers.Bijector] = []
        squeeze: layers.Bijector
        if args.reshape_method == "haar":
            squeeze = layers.HaarDownsampling(input_dim)
        else:
            squeeze = layers.SqueezeLayer(2)

        if not unsqueeze:
            level += [squeeze]

        input_dim *= 4

        level += [_block(args, input_dim) for _ in range(args.level_depth)]

        if unsqueeze:  # when unsqueezing, the unsqueeze layer has to come after the block
            level += [layers.InvertBijector(to_invert=squeeze)]

        level_layer = layers.BijectorChain(level)
        if args.jit:
            level_layer = jit.script(level_layer)
        chain.append(level_layer)
        if i in factor_splits:
            input_dim = round(factor_splits[i] * input_dim)
    return chain


def build_conv_inn(args: InnArgs, input_shape: Tuple[int, ...]) -> layers.Bijector:
    input_dim = input_shape[0]

    full_chain: List[layers.Bijector] = []

    factor_splits = {int(k): float(v) for k, v in args.factor_splits.items()}
    main_chain = _build_multi_scale_chain(args, input_dim, factor_splits)

    if args.oxbow_net:
        up_chain = _build_multi_scale_chain(args, input_dim, factor_splits, unsqueeze=True)
        # TODO: don't flatten here and instead split correctly along the channel axis
        full_chain += [layers.OxbowNet(main_chain, up_chain, factor_splits), layers.Flatten()]
    else:
        full_chain += [layers.FactorOut(main_chain, factor_splits)]

    # flattened_shape = int(product(input_shape))
    # full_chain += [layers.RandomPermutation(flattened_shape)]

    model = layers.BijectorChain(full_chain)

    # if args.jit:
    #     model = jit.script(model)
    return model


def build_discriminator(
    input_shape: Tuple[int, ...],
    target_dim: int,
    train_on_recon: bool,
    frac_enc: float,
    model_fn: ModelFn,
    model_kwargs: Dict[str, Union[float, str, bool]],
    optimizer_kwargs=None,
):
    in_dim = input_shape[0]

    # this is done in models/inn.py
    if not train_on_recon and len(input_shape) > 2:
        in_dim = round(frac_enc * int(product(input_shape)))

    num_classes = target_dim if target_dim > 1 else 2
    discriminator = Classifier(
        model_fn(in_dim, target_dim, **model_kwargs),
        num_classes=num_classes,
        optimizer_kwargs=optimizer_kwargs,
    )

    return discriminator
