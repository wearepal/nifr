"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path

import pandas as pd
import comet_ml  # this import is needed because comet_ml has to be imported before sklearn

# from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
from torch.utils.data.dataset import random_split

from ethicml.algorithms.inprocess.logistic_regression import LR
# from ethicml.algorithms.inprocess.svm import SVM
from ethicml.algorithms.utils import DataTuple  # , PathTuple
from ethicml.evaluators.evaluate_models import run_metrics  # , call_on_saved_data
from ethicml.metrics import Accuracy  # , ProbPos, Theil
from ethicml.data import Adult

from finn.train import main as training_loop
from finn.utils.dataloading import load_dataset, pytorch_data_to_dataframe
from finn.utils.training_utils import (parse_arguments, train_and_evaluate_classifier, encode_dataset,
                                       encode_dataset_no_recon)
from finn.utils.eval_metrics import evaluate_with_classifier, train_zy_head


def main(raw_args=None):
    args = parse_arguments(raw_args)
    whole_train_data, whole_test_data, _, _ = load_dataset(args)

    # shrink test set according to args.data_pcnt
    test_len = int(args.data_pcnt * len(whole_test_data))
    test_data, _ = random_split(whole_test_data,
                                lengths=(test_len, len(whole_test_data) - test_len))

    if args.meta_learn:
        # whole_train_data: D*, whole_val_data: D, whole_test_data: D†
        if args.dataset == 'cmnist':
            whole_train_data.swap_train_test_colorization()
            whole_test_data.swap_train_test_colorization()
            # split the training set to get training and validation sets
            whole_train_data, whole_val_data = random_split(whole_train_data,
                                                            lengths=(50000, 10000))
        else:
            val_len = round(0.1 / 0.75 * len(whole_train_data))
            train_len = len(whole_train_data) - val_len
            whole_train_data, whole_val_data = random_split(whole_train_data,
                                                            lengths=(train_len, val_len))
        # shrink validation set according to args.data_pcnt
        val_len = int(args.data_pcnt * len(whole_val_data))
        val_data, _ = random_split(whole_val_data,
                                   lengths=(val_len, len(whole_val_data) - val_len))
    else:
        val_data = test_data  # just use the test set as validation set

    # shrink train set according to args.data_pcnt
    train_len = int(args.data_pcnt * len(whole_train_data))
    train_data, _ = random_split(whole_train_data,
                                 lengths=(train_len, len(whole_train_data) - train_len))

    if args.dataset == 'cmnist':
        # FIXME: this is a very fragile hack that could break any time
        test_data.palette = whole_test_data.palette

    training_loop(args, train_data, val_data, test_data, log_metrics)


def log_metrics(args, experiment, model, discs, train_data, val_data, test_data):
    """Compute and log a variety of metrics"""
    ethicml_model = LR()

    def _compute_metrics(predictions, actual, name):
        """Compute accuracy and fairness metrics and log them"""
        metrics = run_metrics(predictions, actual, metrics=[Accuracy()], per_sens_metrics=[])
        experiment.log_metric(f"{name} Accuracy", metrics['Accuracy'])
        # experiment.log_metric(f"{name} Theil_Index", metrics['Theil_Index'])
        # experiment.log_metric(f"{name} P(Y=1|s=0)", metrics['prob_pos_sex_Male_0'])
        # experiment.log_metric(f"{name} P(Y=1|s=1)", metrics['prob_pos_sex_Male_1'])
        # experiment.log_metric(f"{name} Theil|s=1", metrics['Theil_Index_sex_Male_1'])
        # experiment.log_metric(f"{name} Theil|s=0", metrics['Theil_Index_sex_Male_0'])x
        # experiment.log_metric(f"{name} Ratio s0/s1", metrics['prob_pos_sex_Male_0/sex_Male_1'])
        # experiment.log_metric(f"{name} Diff s0-s1", metrics['prob_pos_sex_Male_0-sex_Male_1'])
        for key, value in metrics.items():
            print(f"\t\t{key}: {value:.4f}")
        print()  # empty line

    if not args.inv_disc:
        print('Encoding validation set...')
        val_repr = encode_dataset(args, val_data, model)

    if args.meta_learn and not args.inv_disc:
        print('Encoding test set...')
        test_repr = encode_dataset_no_recon(args, test_data, model)
        if args.dataset == 'cmnist':
            ddagger_repr = test_repr['zy']  # s = y
            d_repr = val_repr['zy']  # s independent y
            acc = evaluate_with_classifier(args, ddagger_repr, d_repr, args.zy_dim)
            experiment.log_metric("Meta Accuracy", acc)
            print(f"Meta Accuracy: {acc:.4f}")
        else:
            val_tuple = pytorch_data_to_dataframe(val_data)
            test_tuple = pytorch_data_to_dataframe(test_data)
            ddagger_repr = DataTuple(x=test_repr['zy'], s=test_tuple.s, y=test_tuple.y)
            d_repr = DataTuple(x=val_repr['zy'], s=val_tuple.s, y=val_tuple.y)
            preds_meta = ethicml_model.run(ddagger_repr, d_repr)
            _compute_metrics(preds_meta, d_repr, "Meta")
    if args.inv_disc:
        acc = train_zy_head(args, model, discs, test_data, val_data)
        experiment.log_metric("Meta Accuracy", acc)
        print(f"Accuracy on Ddagger: {acc:.4f}")
        return

    print('Encoding training set...')
    train_repr = encode_dataset(args, train_data, model)

    # if args.dataset == 'cmnist':
    #     # mnist_shape = (-1, 3, 28, 28)
    #
    #     train_x_without_s = pd.DataFrame(np.reshape(np.mean(train_tuple.x, axis=1),
    #                                                 (train_tuple.x.shape[0], -1)))
    #     test_x_without_s = pd.DataFrame(np.reshape(np.mean(test_tuple.x, axis=1),
    #                                                (test_tuple.x.shape[0], -1)))
    #
    #     train_x_with_s = np.reshape(train_tuple.x, (train_tuple.x.shape[0], -1))
    #     test_x_with_s = np.reshape(test_tuple.x, (test_tuple.x.shape[0], -1))
    if args.dataset == 'adult':
        # FIXME: this is needed because the information about feature names got lost
        sens_attrs = Adult().feature_split['s']
        train_tuple = pytorch_data_to_dataframe(train_data, sens_attrs=sens_attrs)
        test_tuple = pytorch_data_to_dataframe(test_data, sens_attrs=sens_attrs)

        train_x_with_s = pd.concat([train_tuple.x, train_tuple.s], axis='columns')
        test_x_with_s = pd.concat([test_tuple.x, test_tuple.s], axis='columns')
        train_x_without_s = train_tuple.x
        test_x_without_s = test_tuple.x

    # model = SVM()
    experiment.log_other("evaluation model", ethicml_model.name)

    # ===========================================================================
    check_originals = True
    if check_originals:
        print("Original x:")

        if args.dataset == 'cmnist':
            print("\tTraining performance")
            clf = train_and_evaluate_classifier(args, train_data, palette=test_data.palette,
                                                pred_s=False, use_s=False)
            preds_x, test_x = clf(train_data)
            _compute_metrics(preds_x, test_x, "Original - Train")

            preds_x, test_x = clf(test_data)
        else:
            train_x = DataTuple(x=train_x_without_s, s=train_tuple.s, y=train_tuple.y)
            test_x = DataTuple(x=test_x_without_s, s=test_tuple.s, y=test_tuple.y)
            preds_x = ethicml_model.run(train_x, test_x)

        print("\tTest performance")
        _compute_metrics(preds_x, test_x, "Original")

        # ===========================================================================
        print("Original x & s:")

        if args.dataset == 'cmnist':
            print("\tTraining performance")
            clf = train_and_evaluate_classifier(args, train_data, palette=test_data.palette,
                                                pred_s=False, use_s=True)
            preds_x_and_s, test_x_and_s = clf(train_data)
            _compute_metrics(preds_x_and_s, test_x_and_s, "Original+s")

            preds_x_and_s, test_x_and_s = clf(test_data)
        else:
            train_x_and_s = DataTuple(train_x_with_s,
                                      s=train_tuple.s,
                                      y=train_tuple.y)
            test_x_and_s = DataTuple(x=test_x_with_s,
                                     s=test_tuple.s,
                                     y=test_tuple.y)
            preds_x_and_s = ethicml_model.run(train_x_and_s, test_x_and_s)

        print("\tTest performance")
        _compute_metrics(preds_x_and_s, test_x_and_s, "Original+s")

    # ===========================================================================
    print("All z:")

    if args.dataset == 'cmnist':
        print("\tTraining performance")
        clf = train_and_evaluate_classifier(args, train_repr['all_z'], palette=test_data.palette,
                                            pred_s=False, use_s=False)
        preds_fair, train_fair = clf(train_repr['all_z'])
        _compute_metrics(preds_fair, train_fair, "All_Z")

        preds_z, test_z = clf(val_repr['all_z'])
    else:
        train_z = DataTuple(x=train_repr['all_z'], s=train_tuple.s, y=train_tuple.y)
        test_z = DataTuple(x=val_repr['all_z'], s=test_tuple.s, y=test_tuple.y)
        preds_z = ethicml_model.run(train_z, test_z)

    print("\tTest performance")
    _compute_metrics(preds_z, test_z, "Z")

    # ===========================================================================
    print("fair:")

    if args.dataset == 'cmnist':
        print("\tTraining performance")
        clf = train_and_evaluate_classifier(args, train_repr['recon_y'], palette=test_data.palette,
                                            pred_s=False, use_s=False)
        preds_fair, train_fair = clf(train_repr['recon_y'])
        _compute_metrics(preds_fair, train_fair, "Fair")

        preds_fair, test_fair = clf(val_repr['recon_y'])
    else:
        train_fair = DataTuple(x=train_repr['zy'], s=train_tuple.s, y=train_tuple.y)
        test_fair = DataTuple(x=val_repr['zy'], s=test_tuple.s, y=test_tuple.y)
        preds_fair = ethicml_model.run(train_fair, test_fair)

    print("\tTest performance")
    _compute_metrics(preds_fair, test_fair, "Fair")

    # ===========================================================================
    print("unfair:")
    if args.dataset == 'cmnist':
        print("\tTraining performance")
        clf = train_and_evaluate_classifier(args, train_repr['recon_s'], palette=test_data.palette,
                                            pred_s=False, use_s=False)
        preds_unfair, train_unfair = clf(train_repr['recon_s'])
        _compute_metrics(preds_unfair, train_unfair, "Unfair")

        preds_unfair, test_unfair = clf(val_repr['recon_s'])
    else:
        train_unfair = DataTuple(x=train_repr['zs'], s=train_tuple.s, y=train_tuple.y)
        test_unfair = DataTuple(x=val_repr['zs'], s=test_tuple.s, y=test_tuple.y)
        preds_unfair = ethicml_model.run(train_unfair, test_unfair)

    print("\tTest performance")
    _compute_metrics(preds_unfair, test_unfair, "Unfair")

    if check_originals:
        # ===========================================================================
        print("predict s from original x:")

        if args.dataset == 'cmnist':
            clf = train_and_evaluate_classifier(args, train_data, palette=test_data.palette,
                                                pred_s=True, use_s=False)
            preds_s_fair, test_fair_predict_s = clf(test_data)
        else:
            train_fair_predict_s = DataTuple(x=train_x_without_s, s=train_tuple.s, y=train_tuple.s)
            test_fair_predict_s = DataTuple(x=test_x_without_s, s=test_tuple.s, y=test_tuple.s)
            preds_s_fair = ethicml_model.run(train_fair_predict_s, test_fair_predict_s)

        results = run_metrics(preds_s_fair, test_fair_predict_s, [Accuracy()], [])
        experiment.log_metric("Fair pred s", results['Accuracy'])
        print(results)

        # ===========================================================================
        print("predict s from original x & s:")

        if args.dataset == 'cmnist':
            clf = train_and_evaluate_classifier(args, train_data, palette=test_data.palette,
                                                pred_s=True, use_s=True)
            preds_s_fair, test_fair_predict_s = clf(test_data)
        else:
            train_fair_predict_s = DataTuple(x=train_x_with_s, s=train_tuple.s, y=train_tuple.s)
            test_fair_predict_s = DataTuple(x=test_x_with_s, s=test_tuple.s, y=test_tuple.s)
            preds_s_fair = ethicml_model.run(train_fair_predict_s, test_fair_predict_s)

        results = run_metrics(preds_s_fair, test_fair_predict_s, [Accuracy()], [])
        experiment.log_metric("Fair pred s", results['Accuracy'])
        print(results)

    # ===========================================================================
    print("predict s from fair representation:")

    if args.dataset == 'cmnist':
        clf = train_and_evaluate_classifier(args, train_repr['recon_y'], palette=test_data.palette,
                                            pred_s=True, use_s=False)
        preds_s_fair, test_fair_predict_s = clf(val_repr['recon_y'])
    else:
        train_fair_predict_s = DataTuple(x=train_repr['zy'], s=train_tuple.s, y=train_tuple.s)
        test_fair_predict_s = DataTuple(x=val_repr['zy'], s=test_tuple.s, y=test_tuple.s)
        preds_s_fair = ethicml_model.run(train_fair_predict_s, test_fair_predict_s)

    results = run_metrics(preds_s_fair, test_fair_predict_s, [Accuracy()], [])
    experiment.log_metric("Fair pred s", results['Accuracy'])
    print(results)

    # ===========================================================================
    print("predict s from unfair representation:")

    if args.dataset == 'cmnist':
        clf = train_and_evaluate_classifier(args, train_repr['recon_s'], palette=test_data.palette,
                                            pred_s=True, use_s=False)
        preds_s_unfair, test_unfair_predict_s = clf(val_repr['recon_s'])
    else:
        train_unfair_predict_s = DataTuple(x=train_repr['zs'], s=train_tuple.s, y=train_tuple.s)
        test_unfair_predict_s = DataTuple(x=val_repr['zs'], s=test_tuple.s, y=test_tuple.s)
        preds_s_unfair = ethicml_model.run(train_unfair_predict_s, test_unfair_predict_s)

    results = run_metrics(preds_s_unfair, test_unfair_predict_s, [Accuracy()], [])
    experiment.log_metric("Unfair pred s", results['Accuracy'])
    print(results)

    # from weight_adjust import main as weight_adjustment
    # weight_adjustment(DataTuple(x=train_all, s=train.s, y=train.y),
    #                   DataTuple(x=test_all, s=test.s, y=test.y),
    #                   train_zs.shape[1])


if __name__ == "__main__":
    main()
