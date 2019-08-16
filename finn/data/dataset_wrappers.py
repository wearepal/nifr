import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from finn.data.adult import grouped_features_indexes


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


