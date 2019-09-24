"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path

import comet_ml
import random

import numpy as np
import torch
from ethicml.algorithms.inprocess.logistic_regression import LR
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from finn.data import DatasetTriplet, get_data_tuples, load_dataset
from finn.optimisation.evaluation import evaluate, encode_dataset
from finn.optimisation.train import main as training_loop
from finn.optimisation.config import parse_arguments
from finn.optimisation.utils import (
    log_images,
)


def random_seed(seed_value, use_cuda) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def log_sample_images(experiment, data, name):
    data_loader = DataLoader(data, shuffle=False, batch_size=64)
    x, s, y = next(iter(data_loader))
    log_images(experiment, x, f"Samples from {name}", prefix='eval')


def log_metrics(args, experiment, model, data, quick_eval=True):
    """Compute and log a variety of metrics"""
    ethicml_model = LR()

    print('Encoding task dataset...')
    task_repr = encode_dataset(args, data.task, model, recon=True, subdir="task")
    print('Encoding task train dataset...')
    task_train_repr = encode_dataset(args, data.task_train, model, recon=True, subdir="task_train")

    repr = DatasetTriplet(
        pretrain=None,
        task=task_repr,
        task_train=task_train_repr,
        input_dim=data.input_dim,
        output_dim=data.output_dim,
    )

    print("===> Predict y from xy")
    evaluate(args, experiment, repr.task_train['x'], repr.task['x'], name='xy', pred_s=False)
    # print("===> Predict s from xy")
    # evaluate(args, experiment, task_train_repr['xy'], task_repr['xy'], name='xy', pred_s=True)

    if quick_eval:
        log_sample_images(experiment, data.task_train, "task_train")
    else:

        if args.dataset == 'adult':
            task_data, task_train_data = get_data_tuples(data.task, data.task_train)
            data = DatasetTriplet(
                pretrain=None,
                task=task_data,
                task_train=task_train_data,
                input_dim=data.input_dim,
                output_dim=data.output_dim,
            )

        experiment.log_other("evaluation model", ethicml_model.name)

        # ===========================================================================

        evaluate(args, experiment, repr.task_train['zy'], repr.task['zy'], name='zy')
        evaluate(args, experiment, repr.task_train['zs'], repr.task['zs'], name='zs')
        evaluate(args, experiment, repr.task_train['xy'], repr.task['xy'], name='xy')
        evaluate(args, experiment, repr.task_train['xs'], repr.task['xs'], name='xs')


def main(raw_args=None) -> None:
    args = parse_arguments(raw_args)
    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)
    datasets = load_dataset(args)
    training_loop(args, datasets, log_metrics)


if __name__ == "__main__":
    main()
