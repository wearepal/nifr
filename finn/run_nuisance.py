"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path

import pandas as pd
import comet_ml  # this import is needed because comet_ml has to be imported before sklearn
# from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
from ethicml.algorithms.inprocess.logistic_regression import LR
# from ethicml.algorithms.inprocess.svm import SVM
from ethicml.algorithms.utils import DataTuple  # , PathTuple
from ethicml.evaluators.evaluate_models import run_metrics  # , call_on_saved_data
from ethicml.metrics import Accuracy, ProbPos, Theil

from finn.train import main as training_loop
from finn.utils.training_utils import parse_arguments, encode_dataset
from finn.utils.dataloading import pytorch_data_to_dataframe, load_adult_data


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


def main():
    # model = ModelWrapper()

    args = parse_arguments()
    train, test, _, _ = load_adult_data(args)
    training_loop(args, train, test, test, log_metrics)


def log_metrics(args, experiment, inn_model, discs, train_data, val_data, _):
    train = pytorch_data_to_dataframe(train_data)
    test = pytorch_data_to_dataframe(val_data)

    train_repr = encode_dataset(args, train_data, inn_model)
    train_all = train_repr['all_z']
    train_zx = train_repr['zy']
    train_zs = train_repr['zs']

    test_repr = encode_dataset(args, val_data, inn_model)
    test_all = test_repr['all_z']
    test_zx = test_repr['zy']
    test_zs = test_repr['zs']

    def _compute_metrics(predictions, actual, name):
        """Compute accuracy and fairness metrics and log them"""
        metrics = run_metrics(predictions, actual, [Accuracy(), Theil()], [ProbPos(), Theil()])
        experiment.log_metric(f"{name} Accuracy", metrics['Accuracy'])
        experiment.log_metric(f"{name} Theil_Index", metrics['Theil_Index'])
        experiment.log_metric(f"{name} P(Y=1|s=0)", metrics['prob_pos_sex_Male_0'])
        experiment.log_metric(f"{name} P(Y=1|s=1)", metrics['prob_pos_sex_Male_1'])
        experiment.log_metric(f"{name} Theil|s=1", metrics['Theil_Index_sex_Male_1'])
        experiment.log_metric(f"{name} Theil|s=0", metrics['Theil_Index_sex_Male_0'])
        experiment.log_metric(f"{name} Ratio s0/s1", metrics['prob_pos_sex_Male_0/sex_Male_1'])
        experiment.log_metric(f"{name} Diff s0-s1", metrics['prob_pos_sex_Male_0-sex_Male_1'])
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")
        print()  # empty line

    model = LR()
    # model = SVM()
    experiment.log_other("evaluation model", model.name)

    print("Original x:")
    train_x = DataTuple(x=train.x, s=train.s, y=train.y)
    test_x = DataTuple(x=test.x, s=test.s, y=test.y)
    preds_x = model.run(train_x, test_x)
    _compute_metrics(preds_x, test_x, "Original")

    print("Original x & s:")
    train_x_and_s = DataTuple(x=pd.concat([train.x, train.s], axis='columns').rename(
        columns={train.s.columns[0]: f"{train.s.columns[0]}_in_x"}), s=train.s, y=train.y)
    test_x_and_s = DataTuple(x=pd.concat([test.x, test.s], axis='columns').rename(
        columns={train.s.columns[0]: f"{train.s.columns[0]}_in_x"}), s=test.s, y=test.y)
    preds_x_and_s = model.run(train_x_and_s, test_x_and_s)
    _compute_metrics(preds_x_and_s, test_x_and_s, "Original+s")

    print("All z:")
    train_z = DataTuple(x=train_all, s=train.s, y=train.y)
    test_z = DataTuple(x=test_all, s=test.s, y=test.y)
    preds_z = model.run(train_z, test_z)
    _compute_metrics(preds_z, test_z, "Z")

    print("nuisance:")
    train_fair = DataTuple(x=train_zx, s=train.s, y=train.y)
    test_fair = DataTuple(x=test_zx, s=test.s, y=test.y)
    preds_fair = model.run(train_fair, test_fair)
    _compute_metrics(preds_fair, test_fair, "Nuisance")

    print("class:")
    train_unfair = DataTuple(x=train_zs, s=train.s, y=train.y)
    test_unfair = DataTuple(x=test_zs, s=test.s, y=test.y)
    preds_unfair = model.run(train_unfair, test_unfair)
    _compute_metrics(preds_unfair, test_unfair, "Class")

    print("predict class from nuisance representation:")
    train_fair_predict_s = DataTuple(x=train_zx, s=train.s, y=train.s)
    test_fair_predict_s = DataTuple(x=test_zx, s=test.s, y=test.s)
    preds_s_fair = model.run(train_fair_predict_s, test_fair_predict_s)
    results = run_metrics(preds_s_fair, test_fair_predict_s, [Accuracy()], [])
    experiment.log_metric("Nuisance predict s", results['Accuracy'])
    print(results)

    print("predict class from class representation:")
    train_unfair_predict_s = DataTuple(x=train_zs, s=train.s, y=train.s)
    test_unfair_predict_s = DataTuple(x=test_zs, s=test.s, y=test.s)
    preds_s_unfair = model.run(train_unfair_predict_s, test_unfair_predict_s)
    results = run_metrics(preds_s_unfair, test_unfair_predict_s, [Accuracy()], [])
    experiment.log_metric("Class pred s", results['Accuracy'])
    print(results)

    # from weight_adjust import main as weight_adjustment
    # weight_adjustment(DataTuple(x=train_all, s=train.s, y=train.y),
    #                   DataTuple(x=test_all, s=test.s, y=test.y),
    #                   train_zs.shape[1])


if __name__ == "__main__":
    main()