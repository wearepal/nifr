import pickle
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm

from finn.data.preprocess_cmnist import get_path_from_args


def update(existing_aggregate, new_value):
    """Upgrade the aggregator to compute the mean and variance online"""
    count, mean, m2 = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    m2 += delta * (new_value - mean)
    return (count, mean, m2)


def finalize(existing_aggregate):
    """Retrieve the mean and variance from an aggregate"""
    (count, mean, m2) = existing_aggregate
    variance = m2 / count
    if count < 2:
        raise ValueError("Cannot compute variance for 0 or 1 elements")
    return mean, variance


class CMNIST(Dataset):
    def __init__(self, args, train=True, normalize=True, normalize_transform=None):
        super().__init__()
        self.train = train
        self.normalize = normalize

        self.path = get_path_from_args(args)

        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)

        if normalize_transform is not None:
            self.normalize_transform = normalize_transform
            return

        if (self.path / "mean_and_std").exists():
            with open(self.path / 'mean_and_std', 'rb') as fp:
                itemlist = pickle.load(fp)
            mean_x, std_x = itemlist[0], itemlist[1]
            print("loaded mean and std from file")
        elif train:
            print("computing mean and std over training set for normalization")

            aggregator = (0, torch.zeros(3), torch.zeros(3))
            for i in tqdm(range(60000)):
                x, _, _ = torch.load(self.path / "train" / str(i))
                aggregator = update(aggregator, x.view(x.size(0), -1).mean(dim=1))

            mean_x, variance_x = finalize(aggregator)
            std_x = variance_x.sqrt()

            itemlist = [mean_x, std_x]
            with open(self.path / "mean_and_std", 'wb') as fp:
                pickle.dump(itemlist, fp)
        else:
            raise ValueError("need to specify the mean and standard deviation")

        self.normalize_transform = torchvision.transforms.Normalize(mean_x, std_x)

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
        return 60000 if self.train else 10000
