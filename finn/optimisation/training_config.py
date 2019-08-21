import argparse


def restricted_float(x):
    x = float(x)
    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def parse_arguments(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['adult', 'cmnist'], default='cmnist')
    parser.add_argument('--data-pcnt', type=restricted_float, metavar='P', default=1.0,
                        help="data %% should be a real value > 0, and up to 1")
    parser.add_argument('--task-mixing-factor', type=float, metavar='P', default=0.0,
                        help='How much of meta train should be mixed into task train?')

    # Colored MNIST settings
    parser.add_argument('--scale', type=float, default=0.02)
    parser.add_argument('-bg', '--background', type=eval, default=False, choices=[True, False])
    parser.add_argument('--black', type=eval, default=True, choices=[True, False])
    parser.add_argument('--binarize', type=eval, default=True, choices=[True, False])
    parser.add_argument('--rotate-data', type=eval, default=False, choices=[True, False])
    parser.add_argument('--shift-data', type=eval, default=False, choices=[True, False])

    parser.add_argument('--root', type=str, default="data")

    parser.add_argument('--base-density', type=str, default='normal', choices=['logistic', 'normal'])
    parser.add_argument('--squeeze-factor', type=int, default=4)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--dims', type=str, default="100-100")
    parser.add_argument('--glow', type=eval, default=True, choices=[True, False])
    parser.add_argument('--batch-norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--bn-lag', type=restricted_float, default=0,
                        help='fraction of current statistics to incorporate into moving average')
    parser.add_argument('--disc-hidden-dims', type=int, default=256)

    parser.add_argument('--early-stopping', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--test-batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--disc-lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-split-seed', type=int, default=888)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save', type=str, default='experiments/finn')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--super-val', type=eval, default=False, choices=[True, False],
                        help='Train classifier on encodings as part of validation step.')
    parser.add_argument('--val-freq', type=int, default=4)
    parser.add_argument('--log-freq', type=int, default=10)

    parser.add_argument('--zs-frac', type=float, default=0.33)
    parser.add_argument('--learn-mask', type=eval, default=True, choices=[True, False],
                        help='whether to learn a mask over the encoding instead of'
                             'using fixed-sized partitions.')
    parser.add_argument('--log-prob-weight', type=float, default=1.0e-3)
    parser.add_argument('--pred-s-weight', type=float, default=1)

    parser.add_argument('--eval-epochs', type=int, metavar='N', default=30)
    parser.add_argument('--eval-lr', type=float, default=1e-3)

    parser.add_argument('--gpu', type=int, default=0, help='which GPU to use (if available)')
    parser.add_argument('--use-comet', type=eval, default=False, choices=[True, False],
                        help='whether to use the comet.ml logging')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Gamma value for Exponential Learning Rate scheduler.')

    parser.add_argument('--pretrain', type=eval, default=True, choices=[True, False],
                        help='Whether to perform unsupervised pre-training.')
    parser.add_argument('--pretrain-pcnt', type=float, default=0.4)
    parser.add_argument('--task-pcnt', type=float, default=0.2)
    parser.add_argument('--drop-native', type=eval, default=True, choices=[True, False])
    parser.add_argument('--drop-discrete', type=eval, default=False)

    parser.add_argument('--save-to-csv', type=eval, default=False, choices=[True, False])

    return parser.parse_args(raw_args)
