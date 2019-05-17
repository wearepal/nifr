"""Definition of the Adult dataset"""
from functools import partial

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from ethicml.data.load import load_data
from ethicml.algorithms.utils import DataTuple, apply_to_joined_tuple, concat_dt
from ethicml.data import Adult
from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.preprocessing.domain_adaptation import domain_split, dataset_from_cond


def load_adult_data(args):
    """Load dataset from the files specified in ARGS and return it as PyTorch datasets"""
    data = load_data(Adult())
    if args.meta_learn:
        select_sy_equal = partial(
            dataset_from_cond,
            cond="(sex_Male == 0 & salary_50K == 0) | (sex_Male == 1 & salary_50K == 1)")
        select_sy_opposite = partial(
            dataset_from_cond,
            cond="(sex_Male == 1 & salary_50K == 0) | (sex_Male == 0 & salary_50K == 1)")
        selected_sy_equal = apply_to_joined_tuple(select_sy_equal, data)
        selected_sy_opposite = apply_to_joined_tuple(select_sy_opposite, data)

        test_tuple, remaining = train_test_split(selected_sy_equal, train_percentage=0.5,
                                                 random_seed=888)
        train_tuple = concat_dt([selected_sy_opposite, remaining], axis='index', ignore_index=True)

        # s and y should not be overly correlated in the training set
        assert train_tuple.s['sex_Male'].corr(train_tuple.y['salary_>50K']) < 0.1
        # but they should be very correlated in the test set
        assert test_tuple.s['sex_Male'].corr(test_tuple.y['salary_>50K']) > 0.99
    elif args.add_sampling_bias:
        train_tuple, test_tuple = domain_split(
            datatup=data,
            tr_cond='education_Masters == 0. & education_Doctorate == 0.',
            te_cond='education_Masters == 1. | education_Doctorate == 1.'
        )
    else:
        train_tuple, test_tuple = train_test_split(data)

    # def load_dataframe(path: Path) -> pd.DataFrame:
    #     """Load dataframe from a parquet file"""
    #     with path.open('rb') as f:
    #         df = pd.read_feather(f)
    #     return torch.tensor(df.values, dtype=torch.float32)
    #
    # train_x = load_dataframe(Path(args.train_x))
    # train_s = load_dataframe(Path(args.train_s))
    # train_y = load_dataframe(Path(args.train_y))
    # test_x = load_dataframe(Path(args.test_x))
    # test_s = load_dataframe(Path(args.test_s))
    # test_y = load_dataframe(Path(args.test_y))

    # train_test_split()
    # train_data = TensorDataset(train_x, train_s, train_y)
    # test_data = TensorDataset(test_x, test_s, test_y)

    scaler = StandardScaler()

    train_scaled = pd.DataFrame(scaler.fit_transform(train_tuple.x), columns=train_tuple.x.columns)
    train_tuple = DataTuple(x=train_scaled, s=train_tuple.s, y=train_tuple.y)
    test_scaled = pd.DataFrame(scaler.transform(test_tuple.x), columns=test_tuple.x.columns)
    test_tuple = DataTuple(x=test_scaled, s=test_tuple.s, y=test_tuple.y)

    train_data = TensorDataset(*[torch.tensor(df.values, dtype=torch.float32) for df in train_tuple])
    test_data = TensorDataset(*[torch.tensor(df.values, dtype=torch.float32) for df in test_tuple])

    return train_data, test_data, train_tuple, test_tuple,


def get_data_tuples(train_data, val_data, test_data):

    # FIXME: this is needed because the information about feature names got lost
    sens_attrs = Adult().feature_split['s']
    train_tuple = pytorch_data_to_dataframe(train_data, sens_attrs=sens_attrs)
    val_tuple = pytorch_data_to_dataframe(val_data, sens_attrs=sens_attrs)
    test_tuple = pytorch_data_to_dataframe(test_data, sens_attrs=sens_attrs)

    return train_tuple, val_tuple, test_tuple


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
