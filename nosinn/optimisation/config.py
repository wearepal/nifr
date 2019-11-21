import argparse

__all__ = ["nosinn_args", "vae_args"]


def restricted_float(x):
    x = float(x)
    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def shared_args(raw_args=None):
    parser = argparse.ArgumentParser()

    # General data set settings
    parser.add_argument("--dataset", choices=["adult", "cmnist", "celeba"], default="cmnist")
    parser.add_argument(
        "--data-pcnt",
        type=restricted_float,
        metavar="P",
        default=1.0,
        help="data %% should be a real value > 0, and up to 1",
    )
    parser.add_argument(
        "--task-mixing-factor",
        type=float,
        metavar="P",
        default=0.0,
        help="How much of meta train should be mixed into task train?",
    )
    parser.add_argument(
        "--pretrain",
        type=eval,
        default=True,
        choices=[True, False],
        help="Whether to perform unsupervised pre-training.",
    )
    parser.add_argument("--pretrain-pcnt", type=float, default=0.4)
    parser.add_argument("--task-pcnt", type=float, default=0.2)

    # Adult data set feature settings
    parser.add_argument("--drop-native", type=eval, default=True, choices=[True, False])
    parser.add_argument("--drop-discrete", type=eval, default=False)

    # Colored MNIST settings
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--greyscale", type=eval, default=False, choices=[True, False])
    parser.add_argument("-bg", "--background", type=eval, default=False, choices=[True, False])
    parser.add_argument("--black", type=eval, default=True, choices=[True, False])
    parser.add_argument("--binarize", type=eval, default=True, choices=[True, False])
    parser.add_argument("--rotate-data", type=eval, default=False, choices=[True, False])
    parser.add_argument("--shift-data", type=eval, default=False, choices=[True, False])
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--quant-level", type=int, default=8, choices=[3, 5, 8])
    parser.add_argument("--input-noise", type=eval, default=True, choices=[True, False])

    # CelebA settings
    parser.add_argument("--celeba-sens-attr", type=str, default="Male", choices=["Male", "Young"])
    parser.add_argument("--celeba-target-attr", type=str, default="Smiling", choices=["Smiling", "Attractive"])

    # Optimization settings
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-split-seed", type=int, default=888)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma value for Exponential Learning Rate scheduler.",
    )
    parser.add_argument(
        "--train-on-recon",
        type=eval,
        default=False,
        choices=[True, False],
        help="whether to train the discriminator on the reconstructions" "of the encodings.",
    )
    parser.add_argument(
        "--recon-detach",
        type=eval,
        default=True,
        choices=[True, False],
        help="Whether to apply the stop gradient operator to the reconstruction.",
    )

    # Evaluation settings
    parser.add_argument("--eval-epochs", type=int, metavar="N", default=40)
    parser.add_argument("--eval-lr", type=float, default=1e-3)

    # Misc
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use (if available)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save", type=str, default="experiments/finn")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--super-val",
        type=eval,
        default=False,
        choices=[True, False],
        help="Train classifier on encodings as part of validation step.",
    )
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--use-wandb", type=eval, choices=[True, False], default=True)
    parser.add_argument(
        "--results-csv", type=str, default="", help="name of CSV file to save results to"
    )
    return parser


def nosinn_args(raw_args=None):
    parser = shared_args(raw_args=raw_args)

    # INN settings
    parser.add_argument(
        "--base-density", type=str, choices=["logistic", "uniform", "normal"], default="normal"
    )
    parser.add_argument(
        "--base-density-std", type=float, default=1, 
        help="Specifies the standard deviation of the base density if a Gaussian."
    )
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--level-depth", type=int, default=3)
    parser.add_argument(
        "--reshape-method", type=str, choices=["squeeze", "haar"], default="squeeze"
    )
    parser.add_argument("--coupling-channels", type=int, default=256)
    parser.add_argument("--coupling-depth", type=int, default=1)
    parser.add_argument("--glow", type=eval, default=True, choices=[True, False])
    parser.add_argument("--batch-norm", type=eval, default=False, choices=[True, False])
    parser.add_argument(
        "--bn-lag",
        type=restricted_float,
        default=0,
        help="fraction of current statistics to incorporate into moving average",
    )
    parser.add_argument("--factor-splits", action=StoreDictKeyPair, nargs="+", default={})
    parser.add_argument("--preliminary-level", type=eval, default=False, choices=[True, False])
    parser.add_argument("--idf", type=eval, default=False, choices=[True, False])
    parser.add_argument(
        "--scaling", choices=["none", "exp", "sigmoid0.5", "add2_sigmoid"], default="sigmoid0.5"
    )
    parser.add_argument("--spectral-norm", type=eval, default=False, choices=[True, False])
    parser.add_argument("--zs-frac", type=float, default=0.02)

    # Auto-encoder settings
    parser.add_argument("--autoencode", type=eval, default=True, choices=[True, False])
    parser.add_argument("--ae-levels", type=int, default=2)
    parser.add_argument("--ae-enc-dim", type=int, default=3)
    parser.add_argument("--ae-channels", type=int, default=64)
    parser.add_argument("--ae-epochs", type=int, default=5)
    parser.add_argument(
        "--ae-loss", type=str, choices=["l1", "l2", "huber", "ce", "mixed"], default="l2"
    )
    parser.add_argument("--ae-loss-weight", type=float, default=1)
    parser.add_argument("--vae", type=eval, choices=[True, False], default=False)
    parser.add_argument("--kl-weight", type=float, default=0.1)

    # Discriminator settings
    parser.add_argument("--disc-lr", type=float, default=3e-4)
    parser.add_argument("--disc-depth", type=int, default=1)
    parser.add_argument("--disc-channels", type=int, default=256)
    parser.add_argument("--disc-hidden-dims", nargs="*", type=int, default=[])

    # Training settings
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--nll-weight", type=float, default=1e-2)
    parser.add_argument("--pred-s-weight", type=float, default=1)
    parser.add_argument("--recon-stability-weight", type=float, default=0)
    parser.add_argument("--gp-weight", type=float, default=0)

    parser.add_argument("--path-to-ae", type=str, default="")

    return parser.parse_args(raw_args)


def vae_args(raw_args=None):
    parser = shared_args(raw_args=raw_args)

    # VAEsettings
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--level-depth", type=int, default=2)
    parser.add_argument("--enc-y-dim", type=int, default=64)
    parser.add_argument("--enc-s-dim", type=int, default=0)
    parser.add_argument("--cond-decoder", type=eval, choices=[True, False], default=True)
    parser.add_argument("--init-channels", type=int, default=32)
    parser.add_argument("--recon-loss", type=str, choices=["l1", "l2", "huber", "ce"], default="l2")
    parser.add_argument("--stochastic", type=eval, choices=[True, False], default=True)
    parser.add_argument("--vgg-weight", type=float, default=0)
    parser.add_argument("--vae", type=eval, choices=[True, False], default=True)

    # Discriminator settings
    parser.add_argument("--disc-enc-y-depth", type=int, default=1)
    parser.add_argument("--disc-enc-y-channels", type=int, default=256)
    parser.add_argument("--disc-enc-s-depth", type=int, default=1)
    parser.add_argument("--disc-enc-s-channels", type=int, default=128)

    # Training settings
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--disc-lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--elbo-weight", type=float, default=1)
    parser.add_argument("--pred-s-weight", type=float, default=1)

    return parser.parse_args(raw_args)


def five_kims_args(raw_args=None):
    parser = shared_args(raw_args=None)
    parser.add_argument("--entropy-weight", type=float, default=0.01)

    # Discriminator settings
    parser.add_argument("--disc-lr", type=float, default=1e-3)
    parser.add_argument("--disc-depth", type=int, default=1)
    parser.add_argument("--disc-channels", type=int, default=256)
    parser.add_argument("--disc-hidden-dims", nargs="*", type=int, default=[])

    # Training settings
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args(raw_args)
