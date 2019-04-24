import argparse
import numpy as np
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', metavar="D", choices=['adult', 'cmnist'], default='cmnist')
    # Colored MNIST settings
    parser.add_argument('--scale', type=float, default=0.02)
    parser.add_argument('--cspace', type=str, default='rgb', choices=['rgb', 'hsv'])
    parser.add_argument('-bg', '--background', type=eval, default=True, choices=[True, False])
    parser.add_argument('--black', type=eval, default=False, choices=[True, False])

    parser.add_argument('--train_x', metavar="PATH")
    parser.add_argument('--train_s', metavar="PATH")
    parser.add_argument('--train_y', metavar="PATH")
    parser.add_argument('--test_x', metavar="PATH")
    parser.add_argument('--test_s', metavar="PATH")
    parser.add_argument('--test_y', metavar="PATH")

    parser.add_argument('--train_new', metavar="PATH")
    parser.add_argument('--test_new', metavar="PATH")

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--dims', type=str, default="100-100")
    parser.add_argument('--nonlinearity', type=str, default="tanh")
    parser.add_argument('--glow', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)

    parser.add_argument('--early_stopping', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--disc_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save', type=str, default='experiments/cnf')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--val_freq', type=int, default=4)
    parser.add_argument('--log_freq', type=int, default=10)

    parser.add_argument('--ind-method', type=str, choices=['hsic', 'disc'], default='disc')
    parser.add_argument('--ind-method2t', type=str, choices=['hsic', 'none'], default='none')

    parser.add_argument('--zs_dim', type=int, default=20)
    parser.add_argument('-iw', '--independence_weight', type=float, default=1.e3)
    parser.add_argument('-iw2t', '--independence_weight_2_towers', type=float, default=1.e3)
    parser.add_argument('--pred_s_weight', type=float, default=1.)
    parser.add_argument('--base_density', default='normal',
                        choices=['normal', 'binormal', 'logitbernoulli', 'bernoulli'])
    parser.add_argument('--base_density_zs', default='',
                        choices=['normal', 'binormal', 'logitbernoulli', 'bernoulli'])

    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use (if available)')
    parser.add_argument('--use_comet', type=eval, default=False, choices=[True, False],
                        help='whether to use the comet.ml logging')
    parser.add_argument('--patience', type=int, default=10, help='Number of iterations without '
                                                                 'improvement in val loss before'
                                                                 'reducing learning rate.')

    return parser.parse_args()


def get_data_dim(data_loader):
    x, _, _ = next(iter(data_loader))
    x_dim = x.size(1)
    x_dim_flat = np.prod(x.shape[1:]).item()

    return x_dim, x_dim_flat


def metameric_sampling(model, xzx, xzs, zs_dim):
    xzx_dim, xzs_dim = xzx.dim(), xzs.dim()

    if xzx_dim == 1 or xzx_dim == 3:
        xzx = xzx.unsqueeze(0)

    if xzs_dim == 1 or xzs_dim == 3:
        xzs = xzs.unsqueeze(0)

    zx = model(xzx)[:, zs_dim]
    zs = model(xzs)[:, zs_dim:]

    zm = torch.cat((zx, zs), dim=1)
    xm = model(zm, reverse=True)

    return xm


def fetch_model(args, x_dim):
    import models

    if args.dataset == 'cmnist':
        model = models.glow(args, x_dim).to(args.device)
    elif args.dataset == 'adult':
        model = models.tabular_model(args, x_dim).to(args.device)
    else:
        raise NotImplementedError("Only works for cmnist and adult - How have you even got"
                                  "hererere?????")
    return model
