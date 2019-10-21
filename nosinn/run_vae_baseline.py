from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ethicml.algorithms.inprocess import LR

from nosinn.data import DatasetTriplet, get_data_tuples, load_dataset
from nosinn.optimisation.evaluation import evaluate, encode_dataset
from nosinn.optimisation.train_vae_baseline import main as training_loop
from nosinn.optimisation.config import vae_args
from nosinn.utils import random_seed


def main(raw_args=None) -> None:
    args = vae_args(raw_args)
    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)
    datasets = load_dataset(args)
    training_loop(args, datasets, None)


if __name__ == "__main__":
    main()
