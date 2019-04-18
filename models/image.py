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

    chain += [layers.Flatten()]
    # chain += [layers.UnsqueezeLayer(upscale_factor=2)]

    if args.base_density == 'bernoulli' or args.base_density_zs == 'bernoulli':
        start_dim = 0 if args.base_density == 'bernoulli' else -args.zs_dim
        if args.base_density_zs == 'bernoulli' or not args.base_density_zs:
            end_dim = None
        else:
            end_dim = -args.zs_dim
        chain.append(layers.SigmoidTransform(start_dim=start_dim, end_dim=end_dim))
    return layers.SequentialFlow(chain)
