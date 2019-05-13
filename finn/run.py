"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path
import comet_ml  # this import is needed because comet_ml has to be imported before sklearn


from ethicml.algorithms.inprocess.logistic_regression import LR
# from ethicml.algorithms.inprocess.svm import SVM

from finn.train import main as training_loop
from finn.utils.dataloading import load_dataset
from finn.utils.evaluate_utils import create_train_test_and_val, make_reprs, metrics_for_meta_learn, get_data_tuples, \
    evaluate_representations
from finn.utils.training_utils import parse_arguments
from finn.utils.eval_metrics import train_zy_head


def main(raw_args=None):
    args = parse_arguments(raw_args)
    whole_train_data, whole_test_data, _, _ = load_dataset(args)
    train_data, val_data, test_data = create_train_test_and_val(args, whole_train_data, whole_test_data)
    training_loop(args, train_data, val_data, test_data, log_metrics)


def log_metrics(args, experiment, model, discs, train_data, val_data, test_data):
    """Compute and log a variety of metrics"""
    ethicml_model = LR()

    train_repr, val_repr, test_repr = make_reprs(args, train_data, val_data, test_data, model)

    reprs = (train_repr, val_repr, test_repr)
    datasets = (train_data, val_data, test_data)

    if args.meta_learn:
        if args.inv_disc:
            acc = train_zy_head(args, model, discs, test_data, val_data)
            experiment.log_metric("Accuracy on Ddagger", acc)
            print(f"Accuracy on Ddagger: {acc:.4f}")
            return
        else:
            metrics_for_meta_learn(args, experiment, ethicml_model, reprs, datasets)

    else:
        if args.inv_disc:
            raise NotImplementedError()
        else:
            pass

    if args.dataset == 'adult':

        train_data, val_data, test_data = get_data_tuples(train_data, val_data, test_data)


    experiment.log_other("evaluation model", ethicml_model.name)

    # ===========================================================================
    check_originals = True
    if check_originals:
        evaluate_representations(args, experiment, train_data, test_data, predict_y=True, use_x=True)
        evaluate_representations(args, experiment, train_data, test_data, predict_y=True, use_x=True, use_s=True)

    # ===========================================================================

    evaluate_representations(args, experiment, train_repr['all_z'], test_repr['all_z'], predict_y=True, use_fair=True, use_unfair=True)
    evaluate_representations(args, experiment, train_repr['zy'], test_repr['zy'], predict_y=True, use_fair=True)
    evaluate_representations(args, experiment, train_repr['zs'], test_repr['zs'], predict_y=True, use_unfair=True)

    # ===========================================================================
    if check_originals:
        evaluate_representations(args, experiment, train_data, test_data, use_x=True)
        evaluate_representations(args, experiment, train_data, test_data, use_s=True, use_x=True)

    # ===========================================================================
    evaluate_representations(args, experiment, train_repr['zy'], test_repr['zy'], use_fair=True)
    evaluate_representations(args, experiment, train_repr['zs'], test_repr['zs'], use_unfair=True)


if __name__ == "__main__":
    main()
