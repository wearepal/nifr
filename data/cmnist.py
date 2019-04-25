from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset


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

    def __init__(self, train=True, normalize=True):
        super().__init__()
        self.train = train
        self.normalize = normalize

        if train and CMNIST.mean is None:
            mean_x = torch.zeros(3)
            std_x = torch.zeros(3)

            for i in range(60000):
                x, s, y = torch.load(Path("./data") / "cmnist" / "train" / str(i))
                mean_x = online_avg(mean_x, i, x.view(x.size(0), -1).mean(dim=1))
                std_x = online_std(mean_x, i, std_x, x.view(x.size(0), -1).mean(dim=1))

            CMNIST.mean = mean_x
            CMNIST.std = std_x

        self.normalize_transform = torchvision.transforms.Normalize(CMNIST.mean, CMNIST.std)

    def __getitem__(self, idx):
        if self.train:
            x, s, y = torch.load(Path("./data") / "cmnist" / "train" / str(idx))
        else:
            x, s, y = torch.load(Path("./data") / "cmnist" / "test" / str(idx))
        if self.normalize:
            x = self.normalize_transform(x)
        return x, s, y

    def __len__(self):
        return 60000 if self.train else 10000
