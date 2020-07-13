"""Definition of the Adult dataset"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ethicml.data import Adult, load_data
from ethicml.preprocessing import (
    domain_split,
    get_biased_and_debiased_subsets,
    get_biased_subset,
    train_test_split,
)
from ethicml.utility import DataTuple
from ethicml.utility.data_helpers import shuffle_df

from .dataset_wrappers import DataTupleDataset

if TYPE_CHECKING:
    from nsfair.configs import SharedArgs


class Triplet(NamedTuple):
    """Small helper class; basically for enabling named returns"""

    meta: Optional[DataTuple]
    task: DataTuple
    task_train: DataTuple


def load_adult_data_tuples(args: SharedArgs) -> Triplet:
    """Load dataset from the files specified in ARGS and return it as PyTorch datasets"""
    adult_dataset = Adult()
    data = load_data(adult_dataset, ordered=True)
    assert data.x.shape[1] == 101

    if args.drop_native:
        data, _, cont_feats = drop_native(data, adult_dataset)
    else:
        cont_feats = adult_dataset.continuous_features

    meta_train: Optional[DataTuple] = None
    if args.pretrain:
        tuples: Triplet = biased_split(args, data)
        assert tuples.meta is not None
        meta_train, task, task_train = tuples.meta, tuples.task, tuples.task_train

    elif args.add_sampling_bias:
        task_train, task = domain_split(
            datatup=data,
            tr_cond="education_Masters == 0. & education_Doctorate == 0.",
            te_cond="education_Masters == 1. | education_Doctorate == 1.",
        )
    else:
        task_train, task = train_test_split(data)

    scaler = StandardScaler()

    task_train_scaled = task_train.x
    task_train_scaled[cont_feats] = scaler.fit_transform(
        task_train.x[cont_feats].to_numpy(np.float32)
    )
    if args.drop_discrete:
        task_train = task_train.replace(x=task_train_scaled[cont_feats])
    else:
        task_train = task_train.replace(x=task_train_scaled)

    task_scaled = task.x
    task_scaled[cont_feats] = scaler.transform(task.x[cont_feats].to_numpy(np.float32))
    if args.drop_discrete:
        task = task.replace(x=task_scaled[cont_feats])
    else:
        task = task.replace(x=task_scaled)

    if args.pretrain:
        assert meta_train is not None
        meta_train_scaled = meta_train.x
        meta_train_scaled[cont_feats] = scaler.transform(
            meta_train.x[cont_feats].to_numpy(np.float32)
        )
        if args.drop_discrete:
            meta_train = meta_train.replace(x=meta_train_scaled[cont_feats])
        else:
            meta_train = meta_train.replace(x=meta_train_scaled)

    return Triplet(meta=meta_train, task=task, task_train=task_train)


def load_adult_data(args) -> Tuple[DataTupleDataset, DataTupleDataset, DataTupleDataset]:
    tuples: Triplet = load_adult_data_tuples(args)
    pretrain_tuple, test_tuple, train_tuple = tuples.meta, tuples.task, tuples.task_train
    assert pretrain_tuple is not None
    source_dataset = Adult()
    disc_features = source_dataset.discrete_features
    cont_features = source_dataset.continuous_features
    pretrain_data = DataTupleDataset(
        pretrain_tuple, disc_features=disc_features, cont_features=cont_features
    )
    train_data = DataTupleDataset(
        train_tuple, disc_features=disc_features, cont_features=cont_features
    )
    test_data = DataTupleDataset(
        test_tuple, disc_features=disc_features, cont_features=cont_features
    )
    return pretrain_data, train_data, test_data


def drop_native(data: DataTuple, adult_dataset: Adult) -> Tuple[DataTuple, List[str], List[str]]:
    """Drop all features that encode the native country except the one for the US"""
    new_x = data.x.drop(
        [
            col
            for col in data.x.columns
            if col.startswith("nat") and col != "native-country_United-States"
        ],
        axis="columns",
    )
    new_x["native-country_not_United-States"] = 1 - new_x["native-country_United-States"]

    disc_feats = adult_dataset.discrete_features
    cont_feats = adult_dataset.continuous_features

    countries = [
        col
        for col in disc_feats
        if (col.startswith("nat") and col != "native-country_United-States")
    ]
    disc_feats = [col for col in disc_feats if col not in countries]
    disc_feats += ["native-country_not_United-States"]
    disc_feats = sorted(disc_feats)

    feats = disc_feats + cont_feats

    data = data.replace(x=new_x[feats])
    assert data.x.shape[1] == 62
    return data, disc_feats, cont_feats


def biased_split(args: SharedArgs, data: DataTuple) -> Triplet:
    """Split the dataset such that the task subset is very biased"""
    use_new_split = True
    if use_new_split:
        task_train_tuple, unbiased = get_biased_subset(
            data=data,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=args.test_pcnt + args.pretrain_pcnt,
            seed=args.data_split_seed,
            data_efficient=True,
        )
    else:
        task_train_tuple, unbiased = get_biased_and_debiased_subsets(
            data=data,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=args.test_pcnt + args.pretrain_pcnt,
            seed=args.data_split_seed,
        )

    task_tuple, meta_tuple = train_test_split(
        unbiased,
        train_percentage=args.test_pcnt / (args.test_pcnt + args.pretrain_pcnt),
        random_seed=args.data_split_seed,
    )
    return Triplet(meta=meta_tuple, task=task_tuple, task_train=task_train_tuple)


def get_data_tuples(*pytorch_datasets):
    """Convert pytorch datasets to datatuples"""
    # FIXME: this is needed because the information about feature names got lost
    sens_attrs = Adult().feature_split["s"]
    return (pytorch_data_to_dataframe(data, sens_attrs=sens_attrs) for data in pytorch_datasets)


def pytorch_data_to_dataframe(dataset, sens_attrs=None):
    """Load a pytorch dataset into a DataTuple consisting of Pandas DataFrames

    Args:
        dataset: PyTorch dataset
        sens_attrs: (optional) list of names of the sensitive attributes
    """
    # create data loader with one giant batch
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # get the data
    data = next(iter(data_loader))
    # convert it to Pandas DataFrames
    data = [pd.DataFrame(tensor.detach().cpu().numpy()) for tensor in data]
    if sens_attrs:
        data[1].columns = sens_attrs
    # create a DataTuple
    return DataTuple(x=data[0], s=data[1], y=data[2])


def shuffle_s(dt: DataTuple) -> DataTuple:
    return dt.replace(s=shuffle_df(dt.s, random_state=42))


def shuffle_y(dt: DataTuple) -> DataTuple:
    return dt.replace(y=shuffle_df(dt.y, random_state=42))
