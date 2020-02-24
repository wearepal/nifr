"""Automatically run evaluation for a given checkpoint

Usage:

    python do_evaluation.py <path to checkpoint>
"""
import sys
import subprocess
from pathlib import Path
import time
from typing import List, Optional

import git
import torch
import tap

from nosinn.configs import CELEBATTRS


class EvalArgs(tap.Tap):
    """Commandline arguments for running evaluation."""

    checkpoint_path: str  # Path to the checkpoint file
    csv_file: Optional[str] = None  # Where to store the results
    eval_id: Optional[int] = None  # ID of the evaluation to run; if not specified, run all.
    gpu: int = 0  # ID of the cuda device to use for evaluation. If negative, run on CPU.
    test_batch_size: int = 1000  # test batch size
    checkout_commit: bool = False  # if True, checkout the commit for the checkpoint
    celeba_target_attr: CELEBATTRS = "Smiling"

    def add_arguments(self):
        self.add_argument("checkpoint_path")  # make the first argument positional


def main():
    # ========================== get checkpoint path and CSV file name ============================
    eval_args = EvalArgs(underscores_to_dashes=True, explicit_bool=True).parse_args()
    chkpt_path = Path(eval_args.checkpoint_path)
    csv_file = eval_args.csv_file if eval_args.csv_file is not None else f"{round(time.time())}.csv"

    # ============================= load ARGS from checkpoint file ================================
    print(f"Loading from '{chkpt_path}' ...")
    chkpt = torch.load(chkpt_path)

    checkout_commit = eval_args.checkout_commit and "sha" in chkpt
    if checkout_commit:
        print("checkout the commit on which the model was trained")
        repo = git.Repo(search_parent_directories=True)
        current_head = repo.head
        repo.git.checkout(chkpt["sha"])

    if "args" in chkpt:
        model_args = chkpt["args"]
    elif "ARGS" in chkpt:
        model_args = chkpt["ARGS"]
    else:
        raise RuntimeError("Checkpoint doesn't contain args.")

    # ================================ prepare values for eval loop ===============================
    dataset = model_args["dataset"]
    if dataset == "cmnist":
        parameter_name = "scale"
        parameter_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    else:
        parameter_name = "task_mixing_factor"
        parameter_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if eval_args.eval_id is not None:
        parameter_values = [parameter_values[eval_args.eval_id]]

    # ============================== construct commandline arguments ==============================
    base_args: List[str] = []
    for key, value in model_args.items():
        key_arg = f"--{key.replace('_', '-')}"
        if isinstance(value, list):
            value_args = [str(item) for item in value]
        elif isinstance(value, dict):
            if value:
                value_args = [f"{item_key}={item_value}" for item_key, item_value in value.items()]
            else:
                continue  # skip this argument and just go to the next one
        else:
            value_args = [str(value)]
        base_args += [key_arg] + value_args

    # ============================== special arguments for evaluation =============================
    base_args += ["--resume", str(chkpt_path.resolve())]
    base_args += ["--evaluate", "True"]
    base_args += ["--results-csv", csv_file]
    base_args += ["--use-wandb", "False"]
    base_args += ["--gpu", str(eval_args.gpu)]
    base_args += ["--test-batch-size", str(eval_args.test_batch_size)]
    base_args += ["--celeba-target-attr", eval_args.celeba_target_attr]

    # ======================================= run eval loop =======================================
    python_exe = sys.executable

    try:
        for parameter_value in parameter_values:
            print(f"Starting run with {parameter_name}: {parameter_value}")
            parameter_args = [f"--{parameter_name.replace('_', '-')}", str(parameter_value)]
            args = [python_exe, "start_nosinn.py"] + base_args + parameter_args
            subprocess.run(args, check=True)

    finally:  # clean up
        if checkout_commit:
            repo.git.checkout(current_head)


if __name__ == "__main__":
    main()
