"""Baseline for Adult dataset"""
import pandas as pd

from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.algorithms.inprocess import LR, SVM, Majority
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.metrics import Accuracy
from ethicml.algorithms.utils import apply_to_joined_tuple, DataTuple

from finn.data.dataloading import load_adult_data


def main():
    class _Namespace:
        meta_learn = True

    args = _Namespace()
    D, D_dagger = load_adult_data(args)
    # D = apply_to_joined_tuple(pd.DataFrame.reset_index, D)
    D = DataTuple(x=D.x.reset_index(drop=True), s=D.s.reset_index(drop=True), y=D.y.reset_index(drop=True))
    D_star, D = train_test_split(D, train_percentage=0.1 / 0.75)
    ethicml_model = SVM(kernel='linear')
    # ethicml_model = Majority()
    predictions = ethicml_model.run(D_dagger, D)
    metrics = run_metrics(predictions, D, metrics=[Accuracy()], per_sens_metrics=[])
    print(f"Accuracy: {metrics['Accuracy']}")


if __name__ == "__main__":
    main()
