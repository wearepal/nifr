import torch

from nosinn.data import load_dataset
from nosinn.optimisation import main_vae as training_loop, vae_args
from nosinn.utils import random_seed


def main(raw_args=None) -> None:
    args = vae_args(raw_args)
    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)
    datasets = load_dataset(args)
    training_loop(args, datasets)


if __name__ == "__main__":
    main()
