import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from finn.data.adult import grouped_features_indexes
from finn.data.misc import set_transform, RandomSampler


class LdAugmentedDataset:
    def __init__(
        self,
        source_dataset,
        ld_augmentations,
        li_augmentation=False,
        base_augmentations: list = None,
        correlation=1.0,
        batch_size=100,
        shuffle=True,
    ):

        self.source_dataset = self._validate_dataset(source_dataset)
        if not 0 <= correlation <= 1:
            raise ValueError("Label-augmentation correlation must be between 0 and 1.")

        self.batch_size = batch_size
        self.shuffle = shuffle

        if not isinstance(ld_augmentations, (list, tuple)):
            ld_augmentations = [ld_augmentations]
        self.ld_augmentations = ld_augmentations

        if base_augmentations is not None:
            base_augmentations = transforms.Compose(base_augmentations)
            set_transform(self.source_dataset, base_augmentations)
        self.base_augmentations = base_augmentations

        self.li_augmentation = li_augmentation
        self.correlation = correlation

        self.dataset = self._validate_dataset(source_dataset)

    def __len__(self):
        return len(self.dataset)

    def num_batches(self):
        return math.ceil(len(self) / self.batch_size)

    def _validate_dataset(self, dataset):
        if isinstance(dataset, DataLoader):
            dataset = dataset
        elif not isinstance(dataset, Dataset):
            raise TypeError("Dataset must be a Dataset or Dataloader object.")

        return dataset

    def subsample(self, pcnt=1.0):
        if not 0 <= pcnt <= 1.0:
            raise ValueError(f"{pcnt} should be in the range (0, 1]")
        num_samples = int(pcnt * len(self.source_dataset))
        inds = list(
            RandomSampler(
                self.source_dataset,
                num_samples=num_samples,
                replacement=False,
            )
        )
        self.inds = inds
        subset = self._sample_from_inds(inds)
        self.dataset = subset

    def _sample_from_inds(self, inds):
        subset = Subset(self.source_dataset, inds)

        return subset

    @staticmethod
    def _validate_data(*args):
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                dtype = torch.long if type(arg) == int else torch.float32
                arg = torch.tensor(arg, dtype=dtype)
            if arg.dim() == 0:
                arg = arg.view(-1)
            yield (arg)

    def __getitem__(self, index):
        return self._subroutine(
            self.dataset.__getitem__(index)
        )

    def _augment(self, x, label):
        for aug in self.ld_augmentations:
            x = aug(x, label)

        return x

    def _subroutine(self, data):

        x, y = data
        s = y
        x, s, y = self._validate_data(x, s, y)

        if self.li_augmentation:
            s = s[torch.randperm(s.nelement())]

        if self.correlation < 1:
            flip_prob = torch.rand(s.shape)
            indexes = flip_prob > self.correlation
            s[indexes] = s[indexes][torch.randperm(indexes.size(0))]

        x = self._augment(x, s)

        if x.dim() == 4 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.squeeze(0)
        s = s.squeeze()
        y = s.squeeze()

        return x, s, y


class DataTupleDataset(Dataset):
    """Wrapper for EthicML datasets"""

    def __init__(self, dataset, source_dataset, transform=None):

        pd.set_option("mode.chained_assignment", None)

        disc_features = source_dataset.discrete_features
        disc_features = [feat for feat in disc_features if feat in dataset.x.columns]
        self.disc_features = disc_features

        cont_features = source_dataset.continuous_features
        cont_features = [feat for feat in cont_features if feat in dataset.x.columns]
        self.cont_features = cont_features
        self.feature_groups = dict(discrete=grouped_features_indexes(self.disc_features))

        self.x_disc = dataset.x[self.disc_features].to_numpy(dtype=np.float32)
        self.x_cont = dataset.x[self.cont_features].to_numpy(dtype=np.float32)
        self.s = dataset.s.to_numpy(dtype=np.float32)
        self.y = dataset.y.to_numpy(dtype=np.float32)

        self._num_samples = dataset.s.shape[0]
        self.transform = transform

    @property
    def transform(self):
        return self.__transform

    @transform.setter
    def transform(self, t):
        t = t or []
        if not isinstance(t, list):
            t = [t]
        self.__transform = t

    def __getitem__(self, index):

        x_disc = self.x_disc[index]
        x_cont = self.x_cont[index]
        s = self.s[index]
        y = self.y[index]

        for tform in self.transform:
            if isinstance(tform, dict):
                if tform["disc"]:
                    x_disc = tform["disc"](x_disc, self.feature_groups)
                if tform["cont"]:
                    x_cont = tform["cont"](x_cont)
            else:
                x_cont = tform(x_cont)
                x_disc = tform(x_disc)

        x = np.concatenate([x_disc, x_cont], axis=0)
        x = torch.from_numpy(x).squeeze(0)
        s = torch.from_numpy(s).squeeze()
        y = torch.from_numpy(y).squeeze()

        return x, s, y

    def __len__(self):
        return self._num_samples


