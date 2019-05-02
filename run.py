"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path

import pandas as pd
import comet_ml  # this import is needed because comet_ml has to be imported before sklearn/torch

# from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
import torch.nn as nn
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from ethicml.algorithms.inprocess.logistic_regression import LR
# from ethicml.algorithms.inprocess.svm import SVM
from ethicml.algorithms.utils import DataTuple  # , PathTuple
from ethicml.evaluators.evaluate_models import run_metrics  # , call_on_saved_data
from ethicml.metrics import Accuracy  # , ProbPos, Theil

from train import current_experiment, main as training_loop
from models import MnistConvNet
from utils.dataloading import load_dataset
from utils.training_utils import (
    parse_arguments, train_and_evaluate_classifier, classifier_training_loop, validate_classifier,
    encode_dataset, encode_dataset_no_recon)


def main():
    args = parse_arguments()
    whole_train_data, whole_test_data, train_tuple, test_tuple = load_dataset(args)

    if args.meta_learn:
        args.zy_frac = 0  # we don't use y here when metalearning
        if args.dataset == 'cmnist':
            whole_train_data.swap_train_test_colorization()
            whole_test_data.swap_train_test_colorization()
        else:
            # something needs to be done to the adult dataset when we're metalearning
            raise RuntimeError("Meta learning doesn't work with adult yet")
        whole_train_dagger = whole_test_data
        # whole_train_data: D*, whole_test_data: D
        whole_train_data, whole_test_data = random_split(whole_train_data, lengths=(50000, 10000))

        dagger_len = int(args.data_pcnt * len(whole_train_dagger))
        dagger_data, _ = random_split(whole_train_dagger,
                                      lengths=(dagger_len, len(whole_train_dagger) - dagger_len))

    train_len = int(args.data_pcnt * len(whole_train_data))
    train_data, _ = random_split(whole_train_data,
                                 lengths=(train_len, len(whole_train_data) - train_len))
    test_len = int(args.data_pcnt * len(whole_test_data))
    test_data, _ = random_split(whole_test_data,
                                lengths=(test_len, len(whole_test_data) - test_len))

    model = training_loop(args, train_data, test_data)
    print('Encoding training set...')
    train_repr = encode_dataset(args, train_data, model)
    print('Encoding test set...')
    test_repr = encode_dataset(args, test_data, model)
    # (train_all, train_zx, train_zs), (test_all, test_zx, test_zs) = training_loop(
    #     args, train_data, test_data)

    experiment = current_experiment()  # works only after training_loop has been called
    experiment.log_dataset_info(name=args.dataset)

    if args.meta_learn:
        print('Encoding dagger set...')
        dagger_repr = encode_dataset_no_recon(args, dagger_data, model)

        meta_clf = MnistConvNet(in_channels=args.zn_dim, out_dims=10, kernel_size=3,
                                hidden_sizes=[256, 256], output_activation=nn.LogSoftmax(dim=1))
        meta_clf = meta_clf.to(args.device)
        classifier_training_loop(args, meta_clf, test_repr['zn'], val_data=dagger_repr['zn'])

        _, acc = validate_classifier(args, meta_clf, dagger_repr['zn'], use_s=True,
                                     pred_s=False, palette=whole_train_dagger.palette)
        experiment.log_metric("Accuracy on Ddagger", acc)
        print(f"Accuracy on Ddagger: {acc:.4f}")
        return
    # flatten the images so that they're amenable to logistic regression

    def _compute_metrics(predictions, actual, name):
        """Compute accuracy and fairness metrics and log them"""
        metrics = run_metrics(predictions, actual, metrics=[Accuracy()], per_sens_metrics=[])
        experiment.log_metric(f"{name} Accuracy", metrics['Accuracy'])
        # experiment.log_metric(f"{name} Theil_Index", metrics['Theil_Index'])
        # experiment.log_metric(f"{name} P(Y=1|s=0)", metrics['prob_pos_sex_Male_0'])
        # experiment.log_metric(f"{name} P(Y=1|s=1)", metrics['prob_pos_sex_Male_1'])
        # experiment.log_metric(f"{name} Theil|s=1", metrics['Theil_Index_sex_Male_1'])
        # experiment.log_metric(f"{name} Theil|s=0", metrics['Theil_Index_sex_Male_0'])
        # experiment.log_metric(f"{name} Ratio s0/s1", metrics['prob_pos_sex_Male_0/sex_Male_1'])
        # experiment.log_metric(f"{name} Diff s0-s1", metrics['prob_pos_sex_Male_0-sex_Male_1'])
        for key, value in metrics.items():
            print(f"\t\t{key}: {value:.4f}")
        print()  # empty line

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
        train_x_with_s = pd.concat([train_tuple.x, train_tuple.s], axis='columns')
        test_x_with_s = pd.concat([test_tuple.x, test_tuple.s], axis='columns')
        train_x_without_s = train_tuple.x
        test_x_without_s = test_tuple.x

    model = LR()
    # model = SVM()
    experiment.log_other("evaluation model", model.name)

    # ===========================================================================
    check_originals = True
    if check_originals:
        print("Original x:")

        if args.dataset == 'cmnist':
            print("\tTraining performance")
            clf = train_and_evaluate_classifier(args, train_data, palette=whole_train_data.palette, pred_s=False,
                                                use_s=False)
            preds_x, test_x = clf(train_data)
            _compute_metrics(preds_x, test_x, "Original - Train")

            preds_x, test_x = clf(test_data)
        else:
            train_x = DataTuple(x=train_x_without_s, s=train_tuple.s, y=train_tuple.y)
            test_x = DataTuple(x=test_x_without_s, s=test_tuple.s, y=test_tuple.y)
            preds_x = model.run(train_x, test_x)

        print("\tTest performance")
        _compute_metrics(preds_x, test_x, "Original")

        # ===========================================================================
        print("Original x & s:")

        if args.dataset == 'cmnist':
            print("\tTraining performance")
            clf = train_and_evaluate_classifier(args, train_data, palette=whole_train_data.palette, pred_s=False,
                                                use_s=True)
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
            preds_x_and_s = model.run(train_x_and_s, test_x_and_s)

        print("\tTest performance")
        _compute_metrics(preds_x_and_s, test_x_and_s, "Original+s")

    # ===========================================================================
    # print("All z:")

    # if args.dataset == 'cmnist':
    #     train_loader = DataLoader(train_all, shuffle=True, batch_size=args.batch_size)
    #     # for data, color, labels in train_loader:
    #     #     save_image(data[:64], './colorized_reconstruction_all.png', nrow=8)
    #     #     shw = torchvision.utils.make_grid(data[:64], nrow=8).permute(1, 2, 0)
    #     #     experiment.log_image(shw, "colorized_reconstruction_all")
    #     #     break

    #     print("\tTraining performance")
    #     clf = run_conv_classifier(args, train_all, palette=whole_train_data.palette, pred_s=False,
    #                               use_s=False)
    #     preds_z, test_z = clf(train_all)
    #     _compute_metrics(preds_z, test_z, "Z")

    #     preds_z, test_z = clf(test_all)
    # else:
    #     train_z = DataTuple(x=train_all, s=train_tuple.s, y=train_tuple.y)
    #     test_z = DataTuple(x=test_all, s=test_tuple.s, y=test_tuple.y)
    #     preds_z = model.run(train_z, test_z)

    # print("\tTest performance")
    # _compute_metrics(preds_z, test_z, "Z")

    # ===========================================================================
    print("fair:")

    if args.dataset == 'cmnist':
        print("\tTraining performance")
        clf = train_and_evaluate_classifier(args, train_repr['recon_y'], palette=whole_train_data.palette, pred_s=False,
                                            use_s=False)
        preds_fair, train_fair = clf(train_repr['recon_y'])
        _compute_metrics(preds_fair, train_fair, "Fair")

        preds_fair, test_fair = clf(test_repr['recon_y'])
    else:
        train_fair = DataTuple(x=train_repr['zy'], s=train_tuple.s, y=train_tuple.y)
        test_fair = DataTuple(x=test_repr['zy'], s=test_tuple.s, y=test_tuple.y)
        preds_fair = model.run(train_fair, test_fair)

    print("\tTest performance")
    _compute_metrics(preds_fair, test_fair, "Fair")

    # ===========================================================================
    print("unfair:")
    if args.dataset == 'cmnist':
        print("\tTraining performance")
        clf = train_and_evaluate_classifier(args, train_repr['recon_s'], palette=whole_train_data.palette, pred_s=False,
                                            use_s=False)
        preds_unfair, train_unfair = clf(train_repr['recon_s'])
        _compute_metrics(preds_unfair, train_unfair, "Unfair")

        preds_unfair, test_unfair = clf(test_repr['recon_s'])
    else:
        train_unfair = DataTuple(x=train_repr['zs'], s=train_tuple.s, y=train_tuple.y)
        test_unfair = DataTuple(x=test_repr['zs'], s=test_tuple.s, y=test_tuple.y)
        preds_unfair = model.run(train_unfair, test_unfair)

    print("\tTest performance")
    _compute_metrics(preds_unfair, test_unfair, "Unfair")

    if check_originals:
        # ===========================================================================
        print("predict s from original x:")

        if args.dataset == 'cmnist':
            clf = run_conv_classifier(args, train_data, palette=whole_train_data.palette, pred_s=True,
                                      use_s=False)
            preds_s_fair, test_fair_predict_s = clf(test_data)
        else:
            train_fair_predict_s = DataTuple(x=train_x_without_s, s=train_tuple.s, y=train_tuple.s)
            test_fair_predict_s = DataTuple(x=test_x_without_s, s=test_tuple.s, y=test_tuple.s)
            preds_s_fair = model.run(train_fair_predict_s, test_fair_predict_s)

        results = run_metrics(preds_s_fair, test_fair_predict_s, [Accuracy()], [])
        experiment.log_metric("Fair pred s", results['Accuracy'])
        print(results)

        # ===========================================================================
        print("predict s from original x & s:")

        if args.dataset == 'cmnist':
            clf = run_conv_classifier(args, train_data, palette=whole_train_data.palette, pred_s=True,
                                      use_s=True)
            preds_s_fair, test_fair_predict_s = clf(test_data)
        else:
            train_fair_predict_s = DataTuple(x=train_x_with_s, s=train_tuple.s, y=train_tuple.s)
            test_fair_predict_s = DataTuple(x=test_x_with_s, s=test_tuple.s, y=test_tuple.s)
            preds_s_fair = model.run(train_fair_predict_s, test_fair_predict_s)

        results = run_metrics(preds_s_fair, test_fair_predict_s, [Accuracy()], [])
        experiment.log_metric("Fair pred s", results['Accuracy'])
        print(results)

    # ===========================================================================
    print("predict s from fair representation:")

    if args.dataset == 'cmnist':
        clf = train_and_evaluate_classifier(args, train_repr['recon_y'], palette=whole_train_data.palette, pred_s=True,
                                            use_s=False)
        preds_s_fair, test_fair_predict_s = clf(test_repr['recon_y'])
    else:
        train_fair_predict_s = DataTuple(x=train_repr['zy'], s=train_tuple.s, y=train_tuple.s)
        test_fair_predict_s = DataTuple(x=test_repr['zy'], s=test_tuple.s, y=test_tuple.s)
        preds_s_fair = model.run(train_fair_predict_s, test_fair_predict_s)

    results = run_metrics(preds_s_fair, test_fair_predict_s, [Accuracy()], [])
    experiment.log_metric("Fair pred s", results['Accuracy'])
    print(results)

    # ===========================================================================
    print("predict s from unfair representation:")

    if args.dataset == 'cmnist':
        clf = train_and_evaluate_classifier(args, train_repr['recon_s'], palette=whole_train_data.palette, pred_s=True,
                                            use_s=False)
        preds_s_unfair, test_unfair_predict_s = clf(test_repr['recon_s'])
    else:
        train_unfair_predict_s = DataTuple(x=train_repr['zs'], s=train_tuple.s, y=train_tuple.s)
        test_unfair_predict_s = DataTuple(x=test_repr['zs'], s=test_tuple.s, y=test_tuple.s)
        preds_s_unfair = model.run(train_unfair_predict_s, test_unfair_predict_s)

    results = run_metrics(preds_s_unfair, test_unfair_predict_s, [Accuracy()], [])
    experiment.log_metric("Unfair pred s", results['Accuracy'])
    print(results)

    # from weight_adjust import main as weight_adjustment
    # weight_adjustment(DataTuple(x=train_all, s=train.s, y=train.y),
    #                   DataTuple(x=test_all, s=test.s, y=test.y),
    #                   train_zs.shape[1])


if __name__ == "__main__":
    main()
