"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path
import comet_ml  # this import is needed because comet_ml has to be imported before sklearn

import random
import numpy as np
import torch

from ethicml.algorithms.inprocess.logistic_regression import LR
# from ethicml.algorithms.inprocess.svm import SVM

from finn.train import main as training_loop
from finn.data import load_dataset
from finn.utils.evaluate_utils import (create_train_test_and_val, metrics_for_meta_learn,
                                       get_data_tuples, evaluate_representations, MetaDataset)
from finn.utils.training_utils import parse_arguments, encode_dataset, encode_dataset_no_recon
from finn.utils.eval_metrics import train_zy_head


def main(raw_args=None):
    args = parse_arguments(raw_args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    whole_train_data, whole_test_data, _, _ = load_dataset(args)
    datasets = create_train_test_and_val(args, whole_train_data, whole_test_data)
    training_loop(args, datasets, log_metrics)


def log_metrics(args, experiment, model, discs, data):
    """Compute and log a variety of metrics"""
    assert args.meta_learn
    ethicml_model = LR()

    # This should be done before computing the representations because computing the representations
    # takes very long and is not needed for this!
    if args.meta_learn and args.inv_disc:
        assert isinstance(data, MetaDataset)
        acc = train_zy_head(args, experiment, model, discs, data.task_train, data.task)
        experiment.log_metric("Meta Accuracy", acc)
        print(f"Meta Accuracy: {acc:.4f}")
        return

    quick_eval = True
    if not args.inv_disc and quick_eval:
        print("Quickly encode task and task train...")
        task_repr = encode_dataset_no_recon(args, data.task, model, recon_zyn=True)
        task_train_repr = encode_dataset_no_recon(args, data.task_train, model, recon_zyn=True)
        evaluate_representations(args, experiment, task_train_repr['recon_yn'], task_repr['recon_yn'],
                                 predict_y=True, use_x=True, use_fair=True, use_s=True)
        return

    print('Encoding task dataset...')
    task_repr_ = encode_dataset(args, data.task, model)
    print('Encoding task train dataset...')
    task_train_repr_ = encode_dataset(args, data.task_train, model)

    repr = MetaDataset(meta_train=None, task=task_repr_, task_train=task_train_repr_)

    if args.meta_learn and not args.inv_disc:
        metrics_for_meta_learn(args, experiment, ethicml_model, repr, data)

    if args.dataset == 'adult':
        _, task_data, task_train_data = get_data_tuples(data.task, data.task, data.task_train)
        data = MetaDataset(meta_train=None, task=task_data, task_train=task_train_data)

    experiment.log_other("evaluation model", ethicml_model.name)

    # ===========================================================================
    check_originals = False
    if check_originals:
        evaluate_representations(args, experiment, data.task_train, data.task,
                                 predict_y=True, use_x=True)
        evaluate_representations(args, experiment, data.task_train, data.task,
                                 predict_y=True, use_x=True, use_s=True)

    # ===========================================================================

    evaluate_representations(args, experiment, repr.task_train['all_z'], repr.task['all_z'],
                             predict_y=True, use_fair=True, use_unfair=True)
    evaluate_representations(args, experiment, repr.task_train['recon_y'], repr.task['recon_y'],
                             predict_y=True, use_x=True, use_fair=True, use_s=True)
    evaluate_representations(args, experiment, repr.task_train['recon_s'], repr.task['recon_s'],
                             predict_y=True, use_x=True, use_unfair=True, use_s=True)

    check_grayscale = False
    if check_grayscale:
        # Grayscale the fair representation
        evaluate_representations(args, experiment, repr.task_train['recon_y'], repr.task['recon_y'],
                                 predict_y=True, use_x=True, use_fair=True)

    # ===========================================================================
    if check_originals:
        evaluate_representations(args, experiment, data.task_train, data.task, use_x=True)
        evaluate_representations(args, experiment, data.task_train, data.task, use_s=True, use_x=True)

    # ===========================================================================
    check_meta_train = False
    if check_meta_train:
        print('Encoding training set...')
        meta_train_repr = encode_dataset(args, data.meta_train, model)
        evaluate_representations(args, experiment, meta_train_repr['zy'], repr.task['zy'], use_fair=True)
        evaluate_representations(args, experiment, meta_train_repr['zs'], repr.task['zs'], use_unfair=True)


if __name__ == "__main__":
    main()
