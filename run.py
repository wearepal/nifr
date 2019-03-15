import sys
from pathlib import Path

from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
from ethicml.algorithms.utils import PathTuple
from ethicml.evaluators.evaluate_models import call_on_saved_data
from ethicml.data import Adult
from ethicml.data.load import load_data
from ethicml.preprocessing.train_test_split import train_test_split


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
    model = ModelWrapper()
    data = load_data(Adult())
    train, test = train_test_split(data)
    call_on_saved_data(model, train, test)


if __name__ == "__main__":
    main()
