"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ethicml.algorithms.inprocess import LR

from finn.data import DatasetTriplet, get_data_tuples, load_dataset
from finn.optimisation.evaluation import evaluate, encode_dataset
from finn.optimisation.train import main as training_loop
from finn.optimisation.config import parse_arguments
from finn.optimisation.utils import log_images
from finn.utils import random_seed


def log_sample_images(data, name, step):
    data_loader = DataLoader(data, shuffle=False, batch_size=64)
    x, _, _ = next(iter(data_loader))
    log_images(x, f"Samples from {name}", prefix="eval", step=step)


def log_metrics(args, model, data, step, quick_eval=True, save_to_csv: Optional[Path] = None):
    """Compute and log a variety of metrics"""
    ethicml_model = LR()

    print("Encoding task dataset...")
    task_repr = encode_dataset(args, data.task, model, recon=True, subdir="task")
    print("Encoding task train dataset...")
    task_train_repr = encode_dataset(args, data.task_train, model, recon=True, subdir="task_train")

    repr = DatasetTriplet(
        pretrain=None,
        task=task_repr,
        task_train=task_train_repr,
        input_dim=data.input_dim,
        output_dim=data.output_dim,
    )

    print("\nComputing metrics...")
    evaluate(
        args,
        step,
        task_train_repr["xy"],
        task_repr["xy"],
        name="xy",
        train_on_recon=True,
        pred_s=False,
        save_to_csv=save_to_csv,
    )
    # print("===> Predict y from xy")
    # evaluate(args, experiment, repr.task_train['x'], repr.task['x'], name='xy', pred_s=False)
    # print("===> Predict s from xy")
    # evaluate(args, experiment, task_train_repr['xy'], task_repr['xy'], name='xy', pred_s=True)

    if quick_eval:
        log_sample_images(data.task_train, "task_train", step=step)
    else:

        if args.dataset == "adult":
            task_data, task_train_data = get_data_tuples(data.task, data.task_train)
            data = DatasetTriplet(
                pretrain=None,
                task=task_data,
                task_train=task_train_data,
                input_dim=data.input_dim,
                output_dim=data.output_dim,
            )

        # ===========================================================================

        evaluate(args, step, repr.task_train["zy"], repr.task["zy"], name="zy")
        evaluate(args, step, repr.task_train["zs"], repr.task["zs"], name="zs")
        evaluate(args, step, repr.task_train["xy"], repr.task["xy"], name="xy")
        evaluate(args, step, repr.task_train["xs"], repr.task["xs"], name="xs")


def main(raw_args=None) -> None:
    args = parse_arguments(raw_args)
    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)
    datasets = load_dataset(args)
    training_loop(args, datasets, log_metrics)


if __name__ == "__main__":
    main()
