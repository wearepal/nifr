"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path

import pandas as pd
import numpy as np
import comet_ml  # this import is needed because comet_ml has to be imported before sklearn

# from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
from ethicml.algorithms.inprocess.logistic_regression import LR
# from ethicml.algorithms.inprocess.svm import SVM
from ethicml.algorithms.utils import DataTuple  # , PathTuple
from ethicml.evaluators.evaluate_models import run_metrics  # , call_on_saved_data
from ethicml.metrics import Accuracy, ProbPos, Theil

from data.preprocess_cmnist import make_cmnist_dataset
from train import current_experiment, main as training_loop
from utils.dataloading import load_dataset
from torch.utils.data.dataset import random_split


# class ModelWrapper(BasicTPA):
#     def __init__(self):
#         super().__init__("fair_invertible", str(Path(__file__).resolve().parent / "train.py"))
#         self.additional_args = sys.argv[1:]

#     def _script_interface(self, train_paths: PathTuple, test_paths: PathTuple, for_train_path,
#                           for_test_path):
#         flags_list = self._path_tuple_to_cmd_args([train_paths, test_paths],
#                                                   ['--train_', '--test_'])

#         # paths to output files
#         flags_list += ['--train_new', str(for_train_path), '--test_new', str(for_test_path)]
#         flags_list += self.additional_args
#         return flags_list
from utils.training_utils import parse_arguments, run_conv_classifier


def main():
    # model = ModelWrapper()

    args = parse_arguments()

    # dataset = Adult()
    # train, test = train_test_split(load_data(dataset))
    #
    train_data, test_data, train_tuple, test_tuple = load_dataset(args)
    train_len = int(args.data_pcnt * len(train_data))
    train_data, _ = random_split(train_data, lengths=(train_len, len(train_data) - train_len))
    test_len = int(args.data_pcnt * len(test_data))
    test_data, _ = random_split(test_data, lengths=(test_len, len(test_data) - test_len))

    (train_all, train_zx, train_zs), (test_all, test_zx, test_zs) \
        = training_loop(args, train_data, test_data)
    experiment = current_experiment()  # works only after training_loop has been called
    experiment.log_dataset_info(name=args.dataset)

    # flatten the images so that they're amenable to logistic regression

    def _compute_metrics(predictions, actual, name):
        """Compute accuracy and fairness metrics and log them"""
        metrics = run_metrics(predictions, actual, metrics=[Accuracy()], per_sens_metrics=[])  #ProbPos(),
        experiment.log_metric(f"{name} Accuracy", metrics['Accuracy'])
        # experiment.log_metric(f"{name} Theil_Index", metrics['Theil_Index'])
        # experiment.log_metric(f"{name} P(Y=1|s=0)", metrics['prob_pos_sex_Male_0'])
        # experiment.log_metric(f"{name} P(Y=1|s=1)", metrics['prob_pos_sex_Male_1'])
        # experiment.log_metric(f"{name} Theil|s=1", metrics['Theil_Index_sex_Male_1'])
        # experiment.log_metric(f"{name} Theil|s=0", metrics['Theil_Index_sex_Male_0'])
        # experiment.log_metric(f"{name} Ratio s0/s1", metrics['prob_pos_sex_Male_0/sex_Male_1'])
        # experiment.log_metric(f"{name} Diff s0-s1", metrics['prob_pos_sex_Male_0-sex_Male_1'])
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")
        print()  # empty line

    if args.dataset == 'cmnist':
        # mnist_shape = (-1, 3, 28, 28)

        train_x_without_s = pd.DataFrame(np.reshape(np.mean(train_tuple.x, axis=1),
                                                    (train_tuple.x.shape[0], -1)))
        test_x_without_s = pd.DataFrame(np.reshape(np.mean(test_tuple.x, axis=1),
                                                   (test_tuple.x.shape[0], -1)))

        train_x_with_s = np.reshape(train_tuple.x, (train_tuple.x.shape[0], -1))
        test_x_with_s = np.reshape(test_tuple.x, (test_tuple.x.shape[0], -1))
    else:
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
        preds_x, test_x = run_conv_classifier(args, train_data, test_data, pred_s=False, use_s=False)
    else:
        train_x = DataTuple(x=train_x_without_s, s=train_tuple.s, y=train_tuple.y)
        test_x = DataTuple(x=test_x_without_s, s=test_tuple.s, y=test_tuple.y)
        preds_x = model.run(train_x, test_x)

    _compute_metrics(preds_x, test_x, "Original")

    # ===========================================================================
    print("Original x & s:")

    if args.dataset == 'cmnist':
        preds_x_and_s, test_x_and_s = run_conv_classifier(args, train_data, test_data, pred_s=False, use_s=True)
    else:
        train_x_and_s = DataTuple(train_x_with_s,
                                  s=train_tuple.s,
                                  y=train_tuple.y)
        test_x_and_s = DataTuple(x=test_x_with_s,
                                 s=test_tuple.s,
                                 y=test_tuple.y)
        preds_x_and_s = model.run(train_x_and_s, test_x_and_s)

    _compute_metrics(preds_x_and_s, test_x_and_s, "Original+s")

    # ===========================================================================
    print("All z:")

    if args.dataset == 'cmnist':
        preds_z, test_z = run_conv_classifier(args, train_all, test_all, pred_s=False, use_s=False)
    else:
        train_z = DataTuple(x=train_all, s=train_tuple.s, y=train_tuple.y)
        test_z = DataTuple(x=test_all, s=test_tuple.s, y=test_tuple.y)
        preds_z = model.run(train_z, test_z)
    _compute_metrics(preds_z, test_z, "Z")

    # ===========================================================================
    print("fair:")

    if args.dataset == 'cmnist':
        preds_fair, test_fair = run_conv_classifier(args, train_zx, test_zx, pred_s=False, use_s=False)
    else:
        train_fair = DataTuple(x=train_zx, s=train_tuple.s, y=train_tuple.y)
        test_fair = DataTuple(x=test_zx, s=test_tuple.s, y=test_tuple.y)
        preds_fair = model.run(train_fair, test_fair)
    _compute_metrics(preds_fair, test_fair, "Fair")

    # ===========================================================================
    print("unfair:")
    if args.dataset == 'cmnist':
        preds_unfair, test_unfair = run_conv_classifier(args, train_zs, test_zs, pred_s=False, use_s=False)
    else:
        train_unfair = DataTuple(x=train_zs, s=train_tuple.s, y=train_tuple.y)
        test_unfair = DataTuple(x=test_zs, s=test_tuple.s, y=test_tuple.y)
        preds_unfair = model.run(train_unfair, test_unfair)
    _compute_metrics(preds_unfair, test_unfair, "Unfair")

    # ===========================================================================
    print("predict s from fair representation:")

    if args.dataset == 'cmnist':
        preds_s_fair, test_fair_predict_s = run_conv_classifier(args, train_zx, test_zx, pred_s=True, use_s=False)
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
        preds_s_unfair, test_unfair_predict_s = run_conv_classifier(args, train_zs, test_zs, pred_s=True, use_s=False)
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
