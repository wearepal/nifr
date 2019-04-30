import layers


def glow(args, input_dim):
    hidden_dims = tuple(map(int, args.dims.split("-")))
    chain = [layers.SqueezeLayer(2)]
    input_dim = input_dim * 2 * 2
    for i in range(args.depth):
        if args.batch_norm:
            chain += [layers.MovingBatchNorm2d(input_dim, bn_lag=args.bn_lag)]
        if args.glow:
            chain += [layers.Invertible1x1Conv(input_dim)]
        chain += [layers.AffineCouplingLayer(input_dim, hidden_dims)]

    chain += [layers.SqueezeLayer(2)]
    # chain += [layers.InvFlatten()]
    # chain += [layers.UnsqueezeLayer(upscale_factor=2)]

    # the layers have to be mixed again:
    chain += [layers.Invertible1x1Conv(input_dim * 2 * 2)]

    return layers.SequentialFlow(chain)
