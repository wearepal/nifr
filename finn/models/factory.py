import numpy as np

from finn import layers
from finn.models.classifier import Classifier


def build_fc_inn(args, input_dim, level_depth: int = None):
    """Build the model with ARGS.depth many layers

    If ARGS.glow is true, then each layer includes 1x1 convolutions.
    """
    level_depth = level_depth or args.level_depth
    _batch_norm = args.batch_norm

    chain = [layers.Flatten()]
    for i in range(level_depth):
        if args.batch_norm:
            chain += [layers.MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag)]
        if args.glow:
            chain += [layers.InvertibleLinear(input_dim)]
        chain += [layers.MaskedCouplingLayer(input_dim,
                                             2 * [args.coupling_dims],
                                             'alternate',
                                             swap=(i % 2 == 0) and not args.glow)]

    chain += [layers.InvertibleLinear(input_dim)]

    return layers.BijectorChain(chain)


def build_conv_inn(args, input_shape):

    input_dim = input_shape[0]

    def _block(_input_dim):
        _chain = []
        if args.idf:
            _chain += [layers.IntegerDiscreteFlow(_input_dim, hidden_channels=args.coupling_channels)]
            _chain += [layers.RandomPermutation(_input_dim)]
        else:
            if args.batch_norm:
                _chain += [layers.MovingBatchNorm2d(_input_dim, bn_lag=args.bn_lag)]
            if args.glow:
                _chain += [layers.Invertible1x1Conv(_input_dim, use_lr_decomp=True)]
            else:
                _chain += [layers.RandomPermutation(_input_dim)]

            if args.no_scaling:
                _chain += [layers.AdditiveCouplingLayer(_input_dim,
                                                        hidden_channels=args.coupling_channels,
                                                        num_blocks=args.coupling_depth,
                                                        pcnt_to_transform=0.25)]
            else:
                _chain += [layers.AffineCouplingLayer(_input_dim,
                                                      num_blocks=args.coupling_depth,
                                                      hidden_channels=args.coupling_channels)]

        return layers.BijectorChain(_chain)

    factor_splits: dict = {int(k): v for k, v in args.factor_splits.items()}
    factor_splits = {int(k): float(v) for k, v in factor_splits.items()}

    chain = []
    for i in range(args.levels):
        level = [layers.SqueezeLayer(2)]
        input_dim *= 4

        level.extend([_block(input_dim) for _ in range(args.level_depth)])

        chain.append(layers.BijectorChain(level))
        if i in factor_splits:
            input_dim = round(factor_splits[i] * input_dim)

    chain = [layers.FactorOut(chain, factor_splits)]

    input_dim = int(np.product(input_shape))
    # if args.idf or not args.glow:
    chain += [layers.RandomPermutation(input_dim)]
    # else:
    #     chain += [layers.InvertibleLinear(input_dim)]

    model = layers.BijectorChain(chain)

    return model


def build_discriminator(args, input_shape, frac_enc,
                        model_fn, model_kwargs,
                        optimizer_args=None):

    in_dim = input_shape[0]

    if not args.train_on_recon and len(input_shape) > 2:
        in_dim = round(frac_enc * int(np.product(input_shape)))

    num_classes = args.y_dim if args.y_dim > 1 else 2
    discriminator = Classifier(
        model_fn(in_dim, args.y_dim, **model_kwargs),
        num_classes=num_classes,
        optimizer_args=optimizer_args
    )

    return discriminator

