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
    input_dim = input_dim * squeeze_factor**2
    for _ in range(args.depth-1):
        chain += [inv_block(args, input_dim, hidden_dims)]

    # if args.multi_head:
    #     # head = partial(inv_block, args=args)
    #     # chain += MultiHead(input_dim,
    #     #                    split_dim=[output_dim, input_dim - (output_dim * 2)],
    #     #                    depths=[1, 2],
    #     #                    layer_fns=head)
    #
    #     heads = []
    #     head_dims = [2 * output_dim, input_dim - (2 * output_dim)]
    #     head_depths = [2, 1]
    #
    #     for depth, head_dim in zip(head_depths, head_dims):
    #         block = [inv_block(args, head_dim, hidden_dims) for _ in range(depth)]
    #         heads += [layers.SequentialFlow(block)]
    #
    #     chain += MultiHead(split_dim=head_dims, head_list=heads)
    chain += [inv_block(args, input_dim, hidden_dims)]

    return layers.SequentialFlow(chain)
