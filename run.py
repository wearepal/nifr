import sys

from pathlib import Path

import pandas as pd

from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
from ethicml.algorithms.inprocess.logistic_regression import LR
from ethicml.algorithms.utils import PathTuple, DataTuple
from ethicml.evaluators.evaluate_models import call_on_saved_data, run_metrics
from ethicml.data import Adult
from ethicml.data.load import load_data
from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.metrics import Accuracy, ProbPos

from train import main as training_loop


class ModelWrapper(BasicTPA):
    def __init__(self):
        super().__init__("fair_invertible", str(Path(__file__).resolve().parent / "train.py"))
        self.additional_args = sys.argv[1:]

    def _script_interface(self, train_paths: PathTuple, test_paths: PathTuple, for_train_path,
                          for_test_path):
        flags_list = self._path_tuple_to_cmd_args([train_paths, test_paths],
                                                  ['--train_', '--test_'])

        # paths to output files
        flags_list += ['--train_new', str(for_train_path), '--test_new', str(for_test_path)]
        flags_list += self.additional_args
        return flags_list


def main():
    # model = ModelWrapper()
    data = load_data(Adult())
    train, test = train_test_split(data)
    (train_all, train_zx, train_zs), (test_all, test_zx, test_zs) = training_loop(train, test)

    lr = LR()

    print("unfair:")
    train_unfair = DataTuple(x=pd.concat([train.x, train.s], axis='columns'), s=train.s, y=train.y)
    test_unfair = DataTuple(x=pd.concat([test.x, test.s], axis='columns'), s=test.s, y=test.y)
    preds_unfair = lr.run(train_unfair, test_unfair)
    results = run_metrics(preds_unfair, test_unfair, [Accuracy()], [ProbPos()])
    print(results)

    print("fair:")
    train_fair = DataTuple(x=train_zx, s=train.s, y=train.y)
    test_fair = DataTuple(x=test_zx, s=test.s, y=test.y)
    preds_fair = lr.run(train_fair, test_fair)
    results = run_metrics(preds_fair, test_fair, [Accuracy()], [ProbPos()])
    print(results)

    print("predict s from fair representation:")
    train_fair_predict_s = DataTuple(x=train_zx, s=train.s, y=train.s)
    test_fair_predict_s = DataTuple(x=test_zx, s=test.s, y=test.s)
    preds_s_fair = lr.run(train_fair_predict_s, test_fair_predict_s)
    results = run_metrics(preds_s_fair, test_fair_predict_s, [Accuracy()], [])
    print(results)

    print("predict s from unfair representation:")
    train_unfair_predict_s = DataTuple(x=train_zs, s=train.s, y=train.s)
    test_unfair_predict_s = DataTuple(x=test_zs, s=test.s, y=test.s)
    preds_s_unfair = lr.run(train_unfair_predict_s, test_unfair_predict_s)
    results = run_metrics(preds_s_unfair, test_unfair_predict_s, [Accuracy()], [])
    print(results)


if __name__ == "__main__":
    main()
