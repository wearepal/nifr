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


def build_conv_inn(args, input_dim):

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
                                                        pcnt_to_transform=0.5)]
            else:
                _chain += [layers.AffineCouplingLayer(_input_dim,
                                                      num_blocks=args.coupling_depth,
                                                      hidden_channels=args.coupling_channels)]

        return layers.BijectorChain(_chain)

    factor_splits = {}
    # factor_splits: dict = args.factor_splits
    # offset = len(chain)
    # factor_splits = {int(k)+offset: float(v) for k, v in factor_splits.items()}

    # input_dim = input_dim_0
    chain = []
    for _ in range(args.levels):
        chain += [layers.SqueezeLayer(2)]
        input_dim = input_dim * 2**2

        for _ in range(args.level_depth):
            chain += [_block(input_dim)]
            # if offset in factor_splits:
            #     input_dim = round(factor_splits[offset] * input_dim)
            # offset += 1

    model = [layers.FactorOut(chain, factor_splits)]
    if args.idf:
        pass
        model += [layers.RandomPermutation(input_dim)]
    else:
        model += [layers.Invertible1x1Conv(input_dim, use_lr_decomp=True)]

    model = layers.BijectorChain(model)

    return model


def build_discriminator(args, input_shape, frac_enc,
                        model_fn, model_kwargs,
                        flatten, optimizer_args=None):

    in_dim = input_shape[0]

    if len(input_shape) > 2:
        h, w = input_shape[1:]
        h += args.padding * 2
        w += args.padding * 2
        if not args.train_on_recon:
            in_dim *= (2**2)**args.levels
            in_dim = round(frac_enc * in_dim)
            h //= args.levels * 2
            w //= args.levels * 2
        if flatten:
            in_dim = int(np.product((in_dim, h, w)))

    num_classes = args.y_dim if args.y_dim > 1 else 2
    discriminator = Classifier(
        model_fn(in_dim, args.y_dim, **model_kwargs),
        num_classes=num_classes,
        optimizer_args=optimizer_args
    )

    return discriminator

