from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from ethicml.evaluators import run_metrics
from ethicml.metrics import Accuracy, ProbPos, TPR
from ethicml.utility import DataTuple

from nosinn.data import load_dataset
from nosinn.models import Classifier
from nosinn.models.configs.classifiers import mp_32x32_net, fc_net, mp_64x64_net
from nosinn.optimisation import get_data_dim
from nosinn.utils import random_seed
from nosinn.configs import SharedArgs


class NaiveArgs(SharedArgs):
    # General data set settings
    greyscale: bool = True

    # Optimization settings
    epochs: int = 40
    test_batch_size: int = 1000
    lr: float = 1e-3

    # Misc settings
    pred_s: bool = False


def main():
    args = NaiveArgs(explicit_bool=True, underscores_to_dashes=True)
    args.parse_args()
    print(args)

    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)

    device = torch.device(f"cuda:{args.gpu}" if use_gpu else "cpu")
    print(f"Running on {device}")

    datasets = load_dataset(args)

    train_data = datasets.task_train
    test_data = datasets.task

    if args.dataset == "cmnist":
        classifier_fn = mp_32x32_net
    elif args.dataset == "adult":

        def adult_fc_net(in_dim, target_dim):
            encoder = fc_net(in_dim, 35, hidden_dims=[35])
            classifier = torch.nn.Linear(35, datasets.y_dim)
            return torch.nn.Sequential(encoder, classifier)

        classifier_fn = adult_fc_net
    else:
        classifier_fn = mp_64x64_net

    train_loader = DataLoader(train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(
        test_data, batch_size=args.test_batch_size, pin_memory=True, shuffle=False
    )

    input_shape = get_data_dim(train_loader)
    target_dim = datasets.s_dim if args.pred_s else datasets.y_dim

    classifier: Classifier = Classifier(
        classifier_fn(input_shape[0], target_dim),
        num_classes=2 if target_dim == 1 else target_dim,
        optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
    )
    classifier.to(device)

    classifier.fit(
        train_loader, test_data=test_loader, epochs=args.epochs, device=device, pred_s=False
    )

    preds, ground_truths, sens = classifier.predict_dataset(test_data, device=device)
    preds = pd.DataFrame(preds, columns=["labels"])
    ground_truths = DataTuple(
        x=pd.DataFrame(sens, columns=["sens"]),
        s=pd.DataFrame(sens, columns=["sens"]),
        y=pd.DataFrame(ground_truths, columns=["labels"]),
    )

    full_name = f"{args.dataset}_naive_baseline"
    if args.dataset == "cmnist":
        full_name += "_greyscale" if args.greyscale else "_color"
    elif args.dataset == "celeba":
        full_name += f"_{args.celeba_sens_attr}"
        full_name += f"_{args.celeba_target_attr}"
    full_name += f"_{str(args.epochs)}epochs"
    metrics = run_metrics(
        preds,
        ground_truths,
        metrics=[Accuracy()],
        per_sens_metrics=[ProbPos(), TPR()] if args.dataset != "cmnist" else [],
    )
    print(f"Results for {full_name}:")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()

    if args.save is not None:
        save_to_csv = Path(args.save_dir)
        save_to_csv.mkdir(exist_ok=True)

        assert isinstance(save_to_csv, Path)
        results_path = save_to_csv / full_name
        if args.dataset == "cmnist":
            value_list = ",".join([str(args.scale)] + [str(v) for v in metrics.values()])
        else:
            value_list = ",".join(
                [str(args.task_mixing_factor)] + [str(v) for v in metrics.values()]
            )
        if results_path.is_file():
            with results_path.open("a") as f:
                f.write(value_list + "\n")
        else:
            with results_path.open("w") as f:
                if args.dataset == "cmnist":
                    f.write(",".join(["Scale"] + [str(k) for k in metrics.keys()]) + "\n")
                else:
                    f.write(",".join(["Mix_fact"] + [str(k) for k in metrics.keys()]) + "\n")
                f.write(value_list + "\n")


if __name__ == "__main__":
    main()
