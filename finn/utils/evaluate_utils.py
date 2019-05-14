from typing import NamedTuple

from ethicml.algorithms.inprocess import LR
from ethicml.algorithms.utils import DataTuple
from ethicml.data import Adult
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.metrics import Accuracy

# from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
from torch.utils.data.dataset import random_split, Dataset
import pandas as pd

from finn.utils.eval_metrics import evaluate_with_classifier
from finn.utils.training_utils import (train_and_evaluate_classifier, encode_dataset,
                                       pytorch_data_to_dataframe)


class MetaDataset(NamedTuple):
    meta_train: Dataset
    task: Dataset
    task_train: Dataset


def create_train_test_and_val(args, whole_train_data, whole_test_data):
    assert args.meta_learn
    # whole_train_data: D*, whole_val_data: D, whole_test_data: D†
    if args.dataset == 'cmnist':
        whole_train_data.swap_train_test_colorization()
        whole_test_data.swap_train_test_colorization()
        # split the training set to get training and validation sets
        whole_train_data, whole_val_data = random_split(whole_train_data, lengths=(50000, 10000))
    else:
        val_len = round(0.1 / 0.75 * len(whole_train_data))
        train_len = len(whole_train_data) - val_len
        whole_train_data, whole_val_data = random_split(whole_train_data, lengths=(train_len, val_len))

    # shrink meta train set according to args.data_pcnt
    meta_train_len = int(args.data_pcnt * len(whole_train_data))
    meta_train_data, _ = random_split(
        whole_train_data, lengths=(meta_train_len, len(whole_train_data) - meta_train_len))

    # shrink task set according to args.data_pcnt
    task_len = int(args.data_pcnt * len(whole_val_data))
    task_data, _ = random_split(whole_val_data, lengths=(task_len, len(whole_val_data) - task_len))

    # shrink task train set according to args.data_pcnt
    task_train_len = int(args.data_pcnt * len(whole_test_data))
    task_train_data, _ = random_split(
        whole_test_data, lengths=(task_train_len, len(whole_test_data) - task_train_len))
    return MetaDataset(meta_train=meta_train_data, task=task_data, task_train=task_train_data)


def get_data_tuples(train_data, val_data, test_data):

    # FIXME: this is needed because the information about feature names got lost
    sens_attrs = Adult().feature_split['s']
    train_tuple = pytorch_data_to_dataframe(train_data, sens_attrs=sens_attrs)
    val_tuple = pytorch_data_to_dataframe(val_data, sens_attrs=sens_attrs)
    test_tuple = pytorch_data_to_dataframe(test_data, sens_attrs=sens_attrs)

    return train_tuple, val_tuple, test_tuple


def compute_metrics(experiment, predictions, actual, name, run_all=False):
    """Compute accuracy and fairness metrics and log them"""
    metrics = run_metrics(predictions, actual, metrics=[Accuracy()], per_sens_metrics=[])
    experiment.log_metric(f"{name} Accuracy", metrics['Accuracy'])
    if run_all:
        experiment.log_metric(f"{name} Theil_Index", metrics['Theil_Index'])
        experiment.log_metric(f"{name} P(Y=1|s=0)", metrics['prob_pos_sex_Male_0'])
        experiment.log_metric(f"{name} P(Y=1|s=1)", metrics['prob_pos_sex_Male_1'])
        experiment.log_metric(f"{name} Theil|s=1", metrics['Theil_Index_sex_Male_1'])
        experiment.log_metric(f"{name} Theil|s=0", metrics['Theil_Index_sex_Male_0'])
        experiment.log_metric(f"{name} Ratio s0/s1", metrics['prob_pos_sex_Male_0/sex_Male_1'])
        experiment.log_metric(f"{name} Diff s0-s1", metrics['prob_pos_sex_Male_0-sex_Male_1'])
    for key, value in metrics.items():
        print(f"\t\t{key}: {value:.4f}")
    print()  # empty line


def metrics_for_meta_learn(args, experiment, clf, repr_tuple, dataset_tuple):
    train_repr, val_repr, test_repr = repr_tuple
    train_data, val_data, test_data = dataset_tuple

    print('Meta Learn Results...')
    if args.dataset == 'cmnist':
        ddagger_repr = test_repr['zy']  # s = y
        d_repr = val_repr['zy']  # s independent y
        acc = evaluate_with_classifier(args, ddagger_repr, d_repr, args.zy_dim)
        experiment.log_metric("Meta Accuracy", acc)
        print(f"Meta Accuracy: {acc:.4f}")
    else:
        val_tuple = pytorch_data_to_dataframe(val_data)
        test_tuple = pytorch_data_to_dataframe(test_data)
        ddagger_repr = test_repr['zy']
        d_repr = val_repr['zy']
        preds_meta = clf.run(ddagger_repr, d_repr)
        compute_metrics(experiment, preds_meta, d_repr, "Meta")


def make_tuple_from_data(train, test, pred_s, use_s):
    if use_s:
        train_x = pd.concat([train.x, train.s], axis='columns')
        test_x = pd.concat([test.x, test.s], axis='columns')
    else:
        train_x = train.x
        test_x = test.x

    if pred_s:
        train_y = train.s
        test_y = test.s
    else:
        train_y = train.y
        test_y = test.y

    return DataTuple(x=train_x, s=train.s, y=train_y), DataTuple(x=test_x, s=test.s, y=test_y)


def get_name(use_s, use_x, predict_y, use_fair, use_unfair):
    name = ""

    if predict_y:
        name += "pred y "
    else:
        name += "pred s "

    if use_x:
        if use_fair:
            name += "Recon Z not S "
        elif use_unfair:
            name += "Recon Z S "
        else:
            name += "Original x "
    else:
        if use_fair and use_unfair:
            name += "all Z "
        elif use_fair:
            name += "Fair Z "
        elif use_unfair:
            name += "Unfair Z "
        else:
            raise NotImplementedError("Not sure how this was reached")
    if use_s:
        name += "& s "
    else:
        name += ""

    return name


def evaluate_representations(args, experiment, train_data, test_data, predict_y=False, use_s=False, use_x=False, use_fair=False, use_unfair=False):

    name = get_name(use_s, use_x, predict_y, use_fair, use_unfair)

    if args.meta_learn:
        if use_fair and use_unfair:
            in_channels = args.zy_dim + args.zs_dim
        elif use_fair:
            in_channels = args.zy_dim
        else:
            in_channels = args.zs_dim
    else:
        if use_fair and use_unfair:
            in_channels = args.zy_dim + args.zs_dim + args.zn_dim
        elif use_fair:
            in_channels = args.zy_dim
        elif use_unfair:
            in_channels = args.zs_dim
        else:
            in_channels = args.zn_dim

    print(f"{name}:")

    if args.dataset == 'cmnist':
        run_all = False
        if use_x:
            clf = train_and_evaluate_classifier(args, train_data, pred_s=not predict_y, use_s=use_s)
        else:
            clf = evaluate_with_classifier(args, train_data, test_data, in_channels=in_channels, pred_s= not predict_y, use_s=use_s, applicative=True)
        preds_x, test_x = clf(test_data=train_data)
        compute_metrics(experiment, preds_x, test_x, f"{name} - Train")
        preds_x, test_x = clf(test_data=test_data)
        print("\tTraining performance")

    else:
        run_all = True
        train_x, test_x = make_tuple_from_data(train_data, test_data, pred_s=not predict_y, use_s=use_s)
        preds_x = LR().run(train_x, test_x)

    print("\tTest performance")
    compute_metrics(experiment, preds_x, test_x, name, run_all=run_all)
