"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path

import random
import numpy as np
import torch
from ethicml.algorithms.inprocess import MLP

from ethicml.algorithms.inprocess.logistic_regression import LR

# from ethicml.algorithms.inprocess.svm import SVM
from torch.utils.data import DataLoader

from finn.optimisation.train import main as training_loop
from finn.data import DatasetTuple, get_data_tuples, load_dataset
from finn.evaluation.evaluate import metrics_for_pretrain, evaluate_representations
from finn.optimisation.training_config import parse_arguments
from finn.optimisation.training_utils import (
    encode_dataset,
    encode_dataset_no_recon,
    log_images,
)


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def main(raw_args=None):
    args = parse_arguments(raw_args)
    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)
    datasets = load_dataset(args)
    training_loop(args, datasets, log_metrics)
    return


def log_sample_images(experiment, data, name):
    data_loader = DataLoader(data, shuffle=False, batch_size=65)
    x, s, y = next(iter(data_loader))
    log_images(experiment, x, f"Samples from {name}", prefix='eval')


def log_metrics(args, experiment, model, discs, data, check_originals=False, save_to_csv=False):
    """Compute and log a variety of metrics"""
    assert args.pretrain
    ethicml_model = LR()

    if check_originals:
        print("Evaluating on original dataset...")
        evaluate_representations(
            args, experiment, data.task_train, data.task, predict_y=True, use_x=True
        )
        if args.dataset == 'cmnist':
            evaluate_representations(
                args, experiment, data.task_train, data.task, predict_y=True, use_x=True, use_s=True
            )

    quick_eval = True  # `quick_eval` disables a lot of evaluations and only runs the most important
    if quick_eval:
        print("Quickly encode task and task train...")
        task_repr = encode_dataset_no_recon(args, data.task, model, recon_zyn=True)
        task_train_repr = encode_dataset_no_recon(args, data.task_train, model, recon_zyn=True)
        log_sample_images(experiment, data.task_train, "task_train")
        evaluate_representations(
            args,
            experiment,
            task_train_repr['recon_yn'],
            task_repr['recon_yn'],
            predict_y=True,
            use_x=True,
            use_fair=True,
            use_s=args.dataset == 'cmnist',
        )
        if args.dataset == 'adult':
            repr_ = DatasetTuple(
                pretrain=None,
                task=task_repr,
                task_train=task_train_repr,
                input_dim=data.input_dim,
                output_dim=data.output_dim,
            )
            metrics_for_pretrain(
                args, experiment, ethicml_model, repr_, data, save_to_csv=save_to_csv
            )
        return

    print('Encoding task dataset...')
    task_repr_ = encode_dataset(args, data.task, model)
    print('Encoding task train dataset...')
    task_train_repr_ = encode_dataset(args, data.task_train, model)

    repr = DatasetTuple(
        task=task_repr_,
        task_train=task_train_repr_,
        input_dim=data.input_dim,
        output_dim=data.output_dim,
    )

    metrics_for_pretrain(args, experiment, ethicml_model, repr, data)

    if args.dataset == 'adult':
        task_data, task_train_data = get_data_tuples(data.task, data.task_train)
        data = DatasetTuple(
            pretrain=None,
            task=task_data,
            task_train=task_train_data,
            input_dim=data.input_dim,
            output_dim=data.output_dim,
        )

    experiment.log_other("evaluation model", ethicml_model.name)

    # ===========================================================================
    if check_originals:
        evaluate_representations(
            args, experiment, data.task_train, data.task, predict_y=True, use_x=True
        )
        evaluate_representations(
            args,
            experiment,
            data.task_train,
            data.task,
            predict_y=True,
            use_x=True,
            use_s=args.dataset == 'cmnist',
        )

    # ===========================================================================

    evaluate_representations(
        args,
        experiment,
        repr.task_train['all_z'],
        repr.task['all_z'],
        predict_y=True,
        use_fair=True,
        use_unfair=True,
    )
    evaluate_representations(
        args,
        experiment,
        repr.task_train['recon_y'],
        repr.task['recon_y'],
        predict_y=True,
        use_x=True,
        use_fair=True,
        use_s=args.dataset == 'cmnist',
    )
    evaluate_representations(
        args,
        experiment,
        repr.task_train['recon_s'],
        repr.task['recon_s'],
        predict_y=True,
        use_x=True,
        use_unfair=True,
        use_s=args.dataset == 'cmnist',
    )

    check_grayscale = False
    if check_grayscale:
        # Grayscale the fair representation
        evaluate_representations(
            args,
            experiment,
            repr.task_train['recon_y'],
            repr.task['recon_y'],
            predict_y=True,
            use_x=True,
            use_fair=True,
        )

    # ===========================================================================
    if check_originals:
        evaluate_representations(args, experiment, data.task_train, data.task, use_x=True)
        evaluate_representations(
            args, experiment, data.task_train, data.task, use_s=True, use_x=True
        )

    # ===========================================================================
    check_pretrain = False
    if check_pretrain:
        print('Encoding training set...')
        pretrain_repr = encode_dataset(args, data.pretrain, model)
        evaluate_representations(
            args, experiment, pretrain_repr['zy'], repr.task['zy'], use_fair=True
        )
        evaluate_representations(
            args, experiment, pretrain_repr['zs'], repr.task['zs'], use_unfair=True
        )


if __name__ == "__main__":
    main()
