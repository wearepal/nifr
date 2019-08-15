from finn import layers
from finn.models.classifier import Classifier


def build_fc_inn(args, input_dim, depth: int = None, batch_norm: bool = None):
    """Build the model with ARGS.depth many layers

    If ARGS.glow is true, then each layer includes 1x1 convolutions.
    """
    _depth = depth or args.depth
    _batch_norm = batch_norm if batch_norm is not None else args.batch_norm

    hidden_dims = tuple(map(int, args.dims.split("-")))
    chain = [layers.InvFlatten()]
    for i in range(_depth):
        if args.batch_norm:
            chain += [layers.MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag)]
        if args.glow and args.dataset == 'adult':
            chain += [layers.InvertibleLinear(input_dim)]
        chain += [layers.MaskedCouplingLayer(input_dim, hidden_dims, 'alternate', swap=i % 2 == 0)]

    chain += [layers.InvertibleLinear(input_dim)]

    return layers.SequentialFlow(chain)


def build_conv_inn(args, input_dim):
    hidden_dims = tuple(map(int, args.dims.split("-")))
    chain = [layers.SqueezeLayer(args.squeeze_factor)]
    input_dim = input_dim * args.squeeze_factor ** 2

    def _inv_block():
        chain = []
        if args.batch_norm:
            chain += [layers.MovingBatchNorm2d(input_dim, bn_lag=args.bn_lag)]
        if args.glow:
            chain += [layers.Invertible1x1Conv(input_dim)]
        chain += [layers.AffineCouplingLayer(input_dim, hidden_dims)]

        return layers.SequentialFlow(chain)

    for _ in range(args.depth):
        chain += [_inv_block()]

    return layers.SequentialFlow(chain)


def build_discriminator(args, input_shape, model_fn):

    in_dim = input_shape[0]

    if not args.learn_mask and len(input_shape) > 2:
        in_dim *= args.squeeze_factor ** 2

    discriminator = Classifier(
        model_fn(in_dim, args.y_dim),
        n_classes=args.y_dim)

    return discriminator

