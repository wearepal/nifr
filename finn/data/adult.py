"""Definition of the Adult dataset"""
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from ethicml.data.load import load_data
from ethicml.algorithms.utils import DataTuple, concat_dt
from ethicml.data import Adult
from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.preprocessing.domain_adaptation import domain_split, query_dt


def load_adult_data(args):
    """Load dataset from the files specified in ARGS and return it as PyTorch datasets"""
    data = load_data(Adult())
    if args.meta_learn:
        sy_equal = query_dt(
            data, "(sex_Male == 0 & salary_50K == 0) | (sex_Male == 1 & salary_50K == 1)")
        sy_opposite = query_dt(
            data, "(sex_Male == 1 & salary_50K == 0) | (sex_Male == 0 & salary_50K == 1)")

        task_train_fraction = 0.5  # how much of sy_equal should be reserved for the task train set
        mix_fact = args.task_mixing_factor  # how much of sy_opp should be mixed into task train set

        sy_equal_task_train, sy_equal_meta_train = train_test_split(
            sy_equal, train_percentage=task_train_fraction * (1 - mix_fact), random_seed=888)
        sy_opp_task_train, sy_opp_meta_train = train_test_split(
            sy_opposite, train_percentage=task_train_fraction * mix_fact, random_seed=888)

        task_train_tuple = concat_dt([sy_equal_task_train, sy_opp_task_train],
                                     axis='index', ignore_index=True)
        meta_train_tuple = concat_dt([sy_equal_meta_train, sy_opp_meta_train],
                                     axis='index', ignore_index=True)

        if mix_fact == 0:
            # s and y should not be overly correlated in the meta train set
            assert abs(meta_train_tuple.s['sex_Male'].corr(meta_train_tuple.y['salary_>50K'])) < 0.1
            # but they should be very correlated in the task train set
            assert task_train_tuple.s['sex_Male'].corr(task_train_tuple.y['salary_>50K']) > 0.99

        # old nomenclature:
        train_tuple, test_tuple = meta_train_tuple, task_train_tuple
    elif args.add_sampling_bias:
        train_tuple, test_tuple = domain_split(
            datatup=data,
            tr_cond='education_Masters == 0. & education_Doctorate == 0.',
            te_cond='education_Masters == 1. | education_Doctorate == 1.'
        )
    else:
        train_tuple, test_tuple = train_test_split(data)

    scaler = StandardScaler()

    test_scaled = pd.DataFrame(scaler.fit_transform(test_tuple.x), columns=test_tuple.x.columns)
    test_tuple = DataTuple(x=test_scaled, s=test_tuple.s, y=test_tuple.y)
    train_scaled = pd.DataFrame(scaler.transform(train_tuple.x), columns=train_tuple.x.columns)
    train_tuple = DataTuple(x=train_scaled, s=train_tuple.s, y=train_tuple.y)

    return train_tuple, test_tuple


def get_data_tuples(*pytorch_datasets):
    """Convert pytorch datasets to datatuples"""
    # FIXME: this is needed because the information about feature names got lost
    sens_attrs = Adult().feature_split['s']
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
