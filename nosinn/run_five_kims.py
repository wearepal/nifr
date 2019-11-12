import torch

from nosinn.data import load_dataset
from nosinn.baselines.five_kims import main as training_loop
from nosinn.optimisation.config import five_kims_args
from nosinn.utils import random_seed


def main(raw_args=None) -> None:
    args = five_kims_args(raw_args)
    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)
    datasets = load_dataset(args)
    training_loop(args, datasets)
