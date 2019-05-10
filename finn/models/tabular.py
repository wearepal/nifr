from finn import layers


def tabular_model(args, input_dim, depth: int=None, batch_norm: bool=None):
    """Build the model with ARGS.depth many layers

    If ARGS.glow is true, then each layer includes 1x1 convolutions.
    """
    _depth = depth or args.depth
    _batch_norm = batch_norm if batch_norm is not None else args.batch_norm

    hidden_dims = tuple(map(int, args.dims.split("-")))
    chain = [layers.InvFlatten()]
    for i in range(_depth):
        if args.glow and args.dataset == 'adult':
            chain += [layers.BruteForceLayer(input_dim)]
        chain += [layers.MaskedCouplingLayer(input_dim, hidden_dims, 'alternate', swap=i % 2 == 0)]
        if _batch_norm:
            chain += [layers.MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag)]
    return layers.SequentialFlow(chain)
