import layers


def tabular_model(args, input_dim):
    """Build the model with ARGS.depth many layers

    If ARGS.glow is true, then each layer includes 1x1 convolutions.
    """
    hidden_dims = tuple(map(int, args.dims.split("-")))
    chain = []
    for i in range(args.depth):
        if args.glow:
            chain += [layers.BruteForceLayer(input_dim)]
        chain += [layers.MaskedCouplingLayer(input_dim, hidden_dims, 'alternate', swap=i % 2 == 0)]
        if args.batch_norm:
            chain += [layers.MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag)]

    if args.base_density == 'bernoulli' or args.base_density_zs == 'bernoulli':
        start_dim = 0 if args.base_density == 'bernoulli' else -args.zs_dim
        if args.base_density_zs == 'bernoulli' or not args.base_density_zs:
            end_dim = None
        else:
            end_dim = -args.zs_dim
        chain.append(layers.SigmoidTransform(start_dim=start_dim, end_dim=end_dim))
    return layers.SequentialFlow(chain)
