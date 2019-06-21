from finn import layers


def inv_block(args, input_dim, hidden_dims):
    chain = []
    if args.batch_norm:
        chain += [layers.MovingBatchNorm2d(input_dim, bn_lag=args.bn_lag)]
    if args.glow:
        chain += [layers.Invertible1x1Conv(input_dim)]
    chain += [layers.AffineCouplingLayer(input_dim, hidden_dims)]

    return layers.SequentialFlow(chain)


def glow(args, input_dim):
    hidden_dims = tuple(map(int, args.dims.split("-")))
    squeeze_factor = 4
    chain = [layers.SqueezeLayer(squeeze_factor)]
    # chain += [layers.SqueezeLayer(2)]
    input_dim = input_dim * squeeze_factor ** 2
    for _ in range(args.depth):
        chain += [inv_block(args, input_dim, hidden_dims)]

    if args.glow:
        chain += [layers.Invertible1x1Conv(input_dim)]
    else:
        chain += [layers.AffineCouplingLayer(input_dim, hidden_dims)]

    return layers.SequentialFlow(chain)
