"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path

import pandas as pd
import comet_ml  # this import is needed because comet_ml has to be imported before sklearn/torch

# from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
import torchvision
from torchvision.utils import save_image
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from ethicml.algorithms.inprocess.logistic_regression import LR
# from ethicml.algorithms.inprocess.svm import SVM
from ethicml.algorithms.utils import DataTuple  # , PathTuple
from ethicml.evaluators.evaluate_models import run_metrics  # , call_on_saved_data
from ethicml.metrics import Accuracy  # , ProbPos, Theil

from train import current_experiment, main as training_loop
from utils.dataloading import load_dataset
from utils.training_utils import parse_arguments, run_conv_classifier


def main():
    args = parse_arguments()

    whole_train_data, whole_test_data, train_tuple, test_tuple = load_dataset(args)
    train_len = int(args.data_pcnt * len(whole_train_data))
    train_data, _ = random_split(whole_train_data,
                                 lengths=(train_len, len(whole_train_data) - train_len))
    test_len = int(args.data_pcnt * len(whole_test_data))
    test_data, _ = random_split(whole_test_data,
                                lengths=(test_len, len(whole_test_data) - test_len))

    (train_all, train_zx, train_zs), (test_all, test_zx, test_zs) = training_loop(
        args, train_data, test_data)
    experiment = current_experiment()  # works only after training_loop has been called
    experiment.log_dataset_info(name=args.dataset)

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

    def _log_images(train_data, test_data, name, nsamples=64, nrows=8, monochrome=False):
        """Make a grid of the given images, save them in a file and log them with Comet"""
        nonlocal experiment
        for data, prefix in [(train_data, "train_"), (test_data, "test_")]:
            dataloader = DataLoader(data, shuffle=False, batch_size=nsamples)
            images, _, _ = next(iter(dataloader))
            if monochrome:
                images = images.mean(dim=1, keepdim=True)
            save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
            shw = torchvision.utils.make_grid(images, nrow=nrows).clamp(0, 1).cpu()
            experiment.log_image(torchvision.transforms.functional.to_pil_image(shw), prefix + name)

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
    print("Original x:")

    if args.dataset == 'cmnist':
        _log_images(train_data, test_data, "colorized_orginal_x_no_s", monochrome=True)

        print("\tTraining performance")
        clf = run_conv_classifier(args, train_data, palette=whole_train_data.palette, pred_s=False,
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
        _log_images(train_data, test_data, "colorized_orginal_x_with_s")

        print("\tTraining performance")
        clf = run_conv_classifier(args, train_data, palette=whole_train_data.palette, pred_s=False,
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
        _log_images(train_zx, test_zx, "colorized_reconstruction_zx")

        print("\tTraining performance")
        clf = run_conv_classifier(args, train_zx, palette=whole_train_data.palette, pred_s=False,
                                  use_s=False)
        preds_fair, test_fair = clf(train_zx)
        _compute_metrics(preds_fair, test_fair, "Fair")

        preds_fair, test_fair = clf(test_zx)
    else:
        train_fair = DataTuple(x=train_zx, s=train_tuple.s, y=train_tuple.y)
        test_fair = DataTuple(x=test_zx, s=test_tuple.s, y=test_tuple.y)
        preds_fair = model.run(train_fair, test_fair)

    print("\tTest performance")
    _compute_metrics(preds_fair, test_fair, "Fair")

    # ===========================================================================
    print("unfair:")
    if args.dataset == 'cmnist':
        _log_images(train_zs, test_zs, "colorized_reconstruction_zs")

        print("\tTraining performance")
        clf = run_conv_classifier(args, train_zs, palette=whole_train_data.palette, pred_s=False,
                                  use_s=False)
        preds_unfair, test_unfair = clf(train_zs)
        _compute_metrics(preds_unfair, test_unfair, "Unfair")

        preds_unfair, test_unfair = clf(test_zs)
    else:
        train_unfair = DataTuple(x=train_zs, s=train_tuple.s, y=train_tuple.y)
        test_unfair = DataTuple(x=test_zs, s=test_tuple.s, y=test_tuple.y)
        preds_unfair = model.run(train_unfair, test_unfair)

    print("\tTest performance")
    _compute_metrics(preds_unfair, test_unfair, "Unfair")

    # ===========================================================================
    print("predict s from fair representation:")

    if args.dataset == 'cmnist':
        clf = run_conv_classifier(args, train_zx, palette=whole_train_data.palette, pred_s=True,
                                  use_s=False)
        preds_s_fair, test_fair_predict_s = clf(test_zx)
    else:
        train_fair_predict_s = DataTuple(x=train_zx, s=train_tuple.s, y=train_tuple.s)
        test_fair_predict_s = DataTuple(x=test_zx, s=test_tuple.s, y=test_tuple.s)
        preds_s_fair = model.run(train_fair_predict_s, test_fair_predict_s)

    results = run_metrics(preds_s_fair, test_fair_predict_s, [Accuracy()], [])
    experiment.log_metric("Fair pred s", results['Accuracy'])
    print(results)

    # ===========================================================================
    print("predict s from unfair representation:")

    if args.dataset == 'cmnist':
        clf = run_conv_classifier(args, train_zs, palette=whole_train_data.palette, pred_s=True,
                                  use_s=False)
        preds_s_unfair, test_unfair_predict_s = clf(test_zs)
    else:
        train_unfair_predict_s = DataTuple(x=train_zs, s=train_tuple.s, y=train_tuple.s)
        test_unfair_predict_s = DataTuple(x=test_zs, s=test_tuple.s, y=test_tuple.s)
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
