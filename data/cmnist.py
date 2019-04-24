from pathlib import Path

import torch
from torch.utils.data import Dataset


class CMNIST(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.train = train

    def __getitem__(self, idx):
        if self.train:
            return torch.load(Path("./data") / "cmnist" / "train" / str(idx))
        else:
            return torch.load(Path("./data") / "cmnist" / "test" / str(idx))

    def __len__(self):
        return 60000 if self.train else 10000
