"""Definition of the Adult dataset"""
from typing import Optional, List, Tuple, NamedTuple
from itertools import groupby

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from ethicml.utility import DataTuple
from ethicml.utility.data_helpers import shuffle_df
from ethicml.data import load_data, Adult
from ethicml.preprocessing import train_test_split
from .adult import drop_native, biased_split, Triplet
from .misc import group_features
from .dataset_wrappers import PerturbedDataTupleDataset, DataTupleDataset


def load_perturbed_adult(args) -> Tuple[DataTupleDataset, DataTupleDataset, DataTupleDataset]:
    """Load dataset from the files specified in ARGS and return it as PyTorch datasets"""
    adult_dataset = Adult()
    data = load_data(adult_dataset, ordered=True)
    assert data.x.shape[1] == 101

    if args.drop_native:
        data, disc_feats, cont_feats = drop_native(data, adult_dataset)
    else:
        disc_feats = adult_dataset.discrete_features
        cont_feats = adult_dataset.continuous_features

    # construct a new x in which all discrete features are "continuous"
    new_x = data.x[cont_feats]
    all_feats = cont_feats
    for name, group in group_features(disc_feats):
        one_hot = data.x[list(group)].to_numpy(np.int64)
        indexes = np.argmax(one_hot, axis=1)
        new_x = new_x.assign(**{name: indexes})
        all_feats.append(name)

    # number of bins
    num_bins = new_x.max().to_numpy() + 1
    # normalize
    new_x = new_x / num_bins

    data = data.replace(x=new_x)

    meta_train: Optional[DataTuple] = None
    if args.pretrain:
        tuples: Triplet = biased_split(args, data)
        meta_train, task, task_train = tuples.meta, tuples.task, tuples.task_train
    else:
        task_train, task = train_test_split(data)

    pretrain_data = PerturbedDataTupleDataset(meta_train, features=all_feats, num_bins=num_bins)
    train_data = PerturbedDataTupleDataset(task_train, features=all_feats, num_bins=num_bins)
    test_data = PerturbedDataTupleDataset(task, features=all_feats, num_bins=num_bins)
    return pretrain_data, train_data, test_data
