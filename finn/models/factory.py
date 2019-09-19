import numpy as np

from finn import layers
from finn.models.classifier import Classifier


def build_fc_inn(args, input_dim, depth: int = None):
    """Build the model with ARGS.depth many layers

    If ARGS.glow is true, then each layer includes 1x1 convolutions.
    """
    _depth = depth or args.depth
    _batch_norm = args.batch_norm

    chain = [layers.Flatten()]
    for i in range(_depth):
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
    hidden_dims = args.coupling_dims
    chain = [layers.SqueezeLayer(args.squeeze_factor)]
    input_dim_0 = input_dim * args.squeeze_factor ** 2

    def _block(_input_dim):
        chain = []
        if args.idf:
            chain += [layers.IntegerDiscreteFlow(_input_dim, hidden_channels=hidden_dims)]
            chain += [layers.RandomPermutation(_input_dim)]
        else:
            if args.batch_norm:
                chain += [layers.MovingBatchNorm2d(_input_dim, bn_lag=args.bn_lag)]
            if args.glow:
                chain += [layers.Invertible1x1Conv(_input_dim, use_lr_decomp=True)]
            else:
                chain += [layers.ReversePermutation(_input_dim)]
            chain += [layers.AffineCouplingLayer(_input_dim, hidden_channels=hidden_dims,
                                                 pcnt_to_transform=0.5)]
            # chain += [layers.AdditiveCouplingLayer(_input_dim, hidden_channels=hidden_dims)]

        return layers.BijectorChain(chain)

    factor_splits: dict = args.factor_splits
    offset = len(chain)
    factor_splits = {k+offset: v for k, v in factor_splits.items()}

    input_dim = input_dim_0
    for _ in range(args.depth):
        chain += [_block(input_dim)]
        if offset in factor_splits:
            input_dim = round(factor_splits[offset] * input_dim)
        offset += 1

    model = [layers.FactorOut(chain, factor_splits)]
    if args.idf:
        pass
        model += [layers.RandomPermutation(input_dim_0)]
    else:
        model += [layers.Invertible1x1Conv(input_dim_0, use_lr_decomp=True)]

    model = layers.BijectorChain(model)

    return model


def build_discriminator(args, input_shape, frac_enc,
                        model_fn, model_kwargs,
                        flatten, optimizer_args=None):

    in_dim = input_shape[0]

    if len(input_shape) > 2:
        h, w = input_shape[1:]
        if not args.train_on_recon:
            in_dim *= args.squeeze_factor ** 2
            in_dim = round(frac_enc * in_dim)
            h //= args.squeeze_factor
            w //= args.squeeze_factor
        if flatten:
            in_dim = int(np.product((in_dim, h, w)))

    num_classes = args.y_dim if args.y_dim > 1 else 2
    discriminator = Classifier(
        model_fn(in_dim, args.y_dim, **model_kwargs),
        num_classes=num_classes,
        optimizer_args=optimizer_args
    )

    return discriminator

