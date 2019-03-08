import sys
from pathlib import Path

from ethicml.algorithms.inprocess.threaded.threaded_in_algorithm import BasicTIA
from ethicml.algorithms.utils import PathTuple
from ethicml.evaluators.evaluate_models import call_on_saved_data
from ethicml.data import Adult
from ethicml.data.load import load_data
from ethicml.preprocessing.train_test_split import train_test_split


class ModelWrapper(BasicTIA):
    def __init__(self):
        super().__init__("fair_invertible", str(Path(__file__).resolve().parent / "train.py"))
        self.additional_args = sys.argv[1:]

    def _script_interface(self, train_paths: PathTuple, test_paths: PathTuple, pred_path):
        flags_list = self._path_tuple_to_cmd_args([train_paths, test_paths],
                                                  ['--train_', '--test_'])

        # paths to output files
        flags_list += ['--predictions', str(pred_path)]
        flags_list += self.additional_args
        return flags_list


def main():
    model = ModelWrapper()
    data = load_data(Adult())
    train, test = train_test_split(data)
    call_on_saved_data(model, train, test)


if __name__ == "__main__":
    main()
