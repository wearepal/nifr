import argparse
from pathlib import Path

import pandas as pd
import torch
from ethicml.evaluators import run_metrics
from ethicml.metrics import Accuracy
from ethicml.utility import DataTuple
from torch.utils.data import DataLoader

from nosinn.data import load_dataset
from nosinn.models import Classifier
from nosinn.models.configs.classifiers import mp_32x32_net, fc_net, mp_64x64_net, resnet_50_ft
from nosinn.optimisation import get_data_dim
from nosinn.utils import random_seed


def parse_arguments():
    parser = argparse.ArgumentParser()
    # General data set settings
    parser.add_argument("--dataset", choices=["adult", "cmnist", "celeba"], default="cmnist")
    parser.add_argument(
        "--data-pcnt",
        type=float,
        metavar="P",
        default=1.0,
        help="data %% should be a real value > 0, and up to 1",
    )
    parser.add_argument(
        "--task-mixing-factor",
        type=float,
        metavar="P",
        default=0.0,
        help="How much of meta train should be mixed into task train.",
    )
    parser.add_argument(
        "--pretrain",
        type=eval,
        default=True,
        choices=[True, False],
        help="Whether to perform unsupervised pre-training.",
    )
    parser.add_argument("--pretrain-pcnt", type=float, default=0.4)
    parser.add_argument("--task-pcnt", type=float, default=0.2)
    parser.add_argument("--data-split-seed", type=int, default=888)

    # Adult data set feature settings
    parser.add_argument("--drop-native", type=eval, default=True, choices=[True, False])
    parser.add_argument("--drop-discrete", type=eval, default=False)

    # Colored MNIST settings
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("-bg", "--background", type=eval, default=False, choices=[True, False])
    parser.add_argument("--black", type=eval, default=True, choices=[True, False])
    parser.add_argument("--binarize", type=eval, default=True, choices=[True, False])
    parser.add_argument("--rotate-data", type=eval, default=False, choices=[True, False])
    parser.add_argument("--shift-data", type=eval, default=False, choices=[True, False])
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--quant-level", type=int, default=8, choices=[3, 5, 8])
    parser.add_argument("--input-noise", type=eval, default=True, choices=[True, False])
    parser.add_argument(
        "--greyscale", type=eval, choices=[True, False], default=True,
        help="Whether to grescale the images. Only applies to coloured MNIST."
    )

    # Optimization settings
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)

    # Misc settings
    parser.add_argument("--pred-s", type=eval, default=False, choices=[True, False])
    parser.add_argument("--root", type=str, default="nosinn/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use (if available)")
    parser.add_argument("--save", type=str, default="finn/baselines/experiments")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)

    args.device = torch.device(f"cuda:{args.gpu}" if use_gpu else "cpu")

    datasets = load_dataset(args)

    train_data = datasets.task_train
    test_data = datasets.task

    if args.dataset == "cmnist":
        classifier_fn = resnet_50_ft
    elif args.dataset == "adult":
        classifier_fn = fc_net
    else:
        classifier_fn = resnet_50_ft

    train_loader = DataLoader(train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(
        test_data, batch_size=args.test_batch_size, pin_memory=True, shuffle=False
    )

    input_shape = get_data_dim(train_loader)
    target_dim = args.s_dim if args.pred_s else args.y_dim
    target_dim = 2 if target_dim == 1 else target_dim

    classifier: Classifier = Classifier(
        classifier_fn(input_shape[0], target_dim),
        num_classes=target_dim,
        optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
    )
    classifier.to(args.device)

    classifier.fit(
        train_loader, test_data=test_loader, epochs=args.epochs, device=args.device, pred_s=False
    )

    preds, ground_truths, sens = classifier.predict_dataset(test_data, device=args.device)
    preds = pd.DataFrame(preds)
    ground_truths = DataTuple(x=None, s=sens, y=pd.DataFrame(ground_truths))

    full_name = f"{args.dataset}_naive_baseline"
    full_name += "_greyscale" if args.greyscale else "_color"
    full_name += "_pred_s" if args.pred_s else "_pred_y"
    metrics = run_metrics(preds, ground_truths, metrics=[Accuracy()], per_sens_metrics=[])
    print(f"Results for {full_name}:")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()

    if args.save is not None:
        save_to_csv = Path(args.save)
        if not save_to_csv.exists():
            save_to_csv.mkdir(exist_ok=True)

        assert isinstance(save_to_csv, Path)
        results_path = save_to_csv / full_name
        value_list = ",".join([str(args.scale)] + [str(v) for v in metrics.values()])
        if results_path.is_file():
            with results_path.open("a") as f:
                f.write(value_list + "\n")
        else:
            with results_path.open("w") as f:
                f.write(",".join(["Scale"] + [str(k) for k in metrics.keys()]) + "\n")
                f.write(value_list + "\n")
