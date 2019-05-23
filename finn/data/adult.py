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
    data = load_data(Adult(), ordered=True)
    assert data.x.shape[1] == 101

    if args.drop_native:
        new_x = data.x.drop(
            [col for col in data.x.columns if col.startswith('nat') and col != "native-country_United-States"], axis=1)
        new_x["native-country_not_United-States"] = (1-new_x["native-country_United-States"])

        disc_feats = Adult().discrete_features
        cont_feats = Adult().continuous_features

        countries = [col for col in disc_feats if (col.startswith('nat') and col != "native-country_United-States")]
        disc_feats = [col for col in disc_feats if col not in countries]
        disc_feats += ["native-country_not_United-States"]
        disc_feats = sorted(disc_feats)

        feats = disc_feats + cont_feats

        data = DataTuple(x=new_x[feats], s=data.s, y=data.y)
        assert data.x.shape[1] == 62

    if args.meta_learn:
        def _random_split(data, first_pcnt):
            return train_test_split(
                data, train_percentage=first_pcnt, random_seed=args.data_split_seed)

        not_task, task = _random_split(data, first_pcnt=1 - args.task_pcnt)
        meta, not_task_or_meta = _random_split(not_task, first_pcnt=args.meta_pcnt / (1 - args.task_pcnt))

        sy_equal = query_dt(
            not_task_or_meta, "(sex_Male == 0 & salary_50K == 0) | (sex_Male == 1 & salary_50K == 1)")
        sy_opposite = query_dt(
            not_task_or_meta, "(sex_Male == 1 & salary_50K == 0) | (sex_Male == 0 & salary_50K == 1)")

        # task_train_fraction = 1.  # how much of sy_equal should be reserved for the task train set
        mix_fact = args.task_mixing_factor  # how much of sy_opp should be mixed into task train set

        sy_equal_task_train, _ = _random_split(sy_equal, first_pcnt=(1 - mix_fact))
        sy_opp_task_train, _ = _random_split(sy_opposite, first_pcnt=mix_fact)

        task_train_tuple = concat_dt([sy_equal_task_train, sy_opp_task_train],
                                     axis='index', ignore_index=True)
        # meta_train_tuple = concat_dt([sy_equal_meta_train, sy_opp_meta_train],
        #                              axis='index', ignore_index=True)

        if mix_fact == 0:
            # s & y should be very correlated in the task train set
            assert task_train_tuple.s['sex_Male'].corr(task_train_tuple.y['salary_>50K']) > 0.99

        # old nomenclature:
        meta_train = meta
        task = task
        task_train = task_train_tuple

    elif args.add_sampling_bias:
        meta_train = None
        task_train, task = domain_split(
            datatup=data,
            tr_cond='education_Masters == 0. & education_Doctorate == 0.',
            te_cond='education_Masters == 1. | education_Doctorate == 1.'
        )
    else:
        task_train, task = train_test_split(data)
        meta_train = None

    scaler = StandardScaler()

    cont_feats = Adult().continuous_features

    task_train_scaled = task_train.x
    task_train_scaled[cont_feats] = scaler.fit_transform(task_train.x[cont_feats])
    task_train = DataTuple(x=task_train_scaled, s=task_train.s, y=task_train.y)

    task_scaled = task.x
    task_scaled[cont_feats] = scaler.transform(task.x[cont_feats])
    task = DataTuple(x=task_scaled, s=task.s, y=task.y)

    if args.meta_learn:
        meta_train_scaled = meta_train.x
        meta_train_scaled[cont_feats] = scaler.transform(meta_train.x[cont_feats])
        meta_train = DataTuple(x=meta_train_scaled, s=meta_train.s, y=meta_train.y)

    return meta_train, task, task_train


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
