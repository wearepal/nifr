import os
import pickle
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm

from data.preprocess_cmnist import dataset_args_to_str, get_path_from_args
from utils import utils


def online_avg(last_avg, last_N, new_val):
    return ((last_avg*last_N)+new_val)/(last_N+1)


def online_std(last_avg, last_N, last_std, new_val):
    if last_N == 0:
        return 0
    new_avg = online_avg(last_avg, last_N, new_val)
    new_std = last_std + (new_val - last_avg)*(new_val - new_avg)
    return new_std


class CMNIST(Dataset):

    mean = None
    std = None

    def __init__(self, args, train=True, normalize=True):
        super().__init__()
        self.train = train
        self.normalize = normalize

        self.path = get_path_from_args(args)

        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        LOGGER = utils.get_logger(logpath=save_dir / 'logs', filepath=Path(__file__).resolve())

        if CMNIST.mean is None and os.path.exists(self.path / "mean_and_std"):
            with open(self.path / 'mean_and_std', 'rb') as fp:
                itemlist = pickle.load(fp)
            CMNIST.mean = itemlist[0]
            CMNIST.std = itemlist[1]
            LOGGER.info("loaded mean and std from file")

        if train and CMNIST.mean is None:
            LOGGER.info("computing mean and std over training set for normalization")

            mean_x = torch.zeros(3)
            std_x = torch.zeros(3)

            for i in tqdm(range(60000)):
                x, s, y = torch.load(self.path / "train" / str(i))
                mean_x = online_avg(mean_x, i, x.view(x.size(0), -1).mean(dim=1))
                std_x = online_std(mean_x, i, std_x, x.view(x.size(0), -1).mean(dim=1))

            CMNIST.mean = mean_x
            CMNIST.std = std_x

            itemlist = [mean_x, std_x]
            with open(self.path / "mean_and_std", 'wb') as fp:
                pickle.dump(itemlist, fp)

        self.normalize_transform = torchvision.transforms.Normalize(CMNIST.mean, CMNIST.std)

    def __getitem__(self, idx):
        dataset = "train" if self.train else "test"
        if isinstance(idx, int):
            x, s, y = torch.load(self.path / dataset / str(idx))
        elif isinstance(idx, torch.Tensor):
            x, s, y = torch.load(self.path / dataset / str(idx.item()))
        else: raise NotImplementedError("index must be an int or a tensor")

        if self.normalize:
            x = self.normalize_transform(x)
        return x, s, y

    def __len__(self):
        return 60 if self.train else 10
