from pathlib import Path
from typing import Optional, Dict
from argparse import Namespace
import pandas as pd
import wandb

from ethicml.algorithms.inprocess import LR
from ethicml.evaluators import run_metrics
from ethicml.metrics import Accuracy, Theil, ProbPos, TPR, TNR, PPV, NMI
from ethicml.utility import DataTuple

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from nosinn.data import get_data_tuples
from nosinn.models.classifier import Classifier
from nosinn.models.configs.classifiers import fc_net, mp_32x32_net, mp_64x64_net
from nosinn.models.inn import BipartiteInn


def compute_metrics(predictions, actual, name, step, run_all=False) -> Dict[str, float]:
    """Compute accuracy and fairness metrics and log them"""

    if run_all:
        metrics = run_metrics(
            predictions,
            actual,
            metrics=[Accuracy(), Theil(), TPR(), TNR(), PPV(), NMI(base="y"), NMI(base="s")],
            per_sens_metrics=[
                Theil(),
                ProbPos(),
                TPR(),
                TNR(),
                PPV(),
                NMI(base="y"),
                NMI(base="s"),
            ],
        )
        wandb.log({f"{name} Accuracy": metrics["Accuracy"]}, step=step)
        wandb.log({f"{name} TPR": metrics["TPR"]}, step=step)
        wandb.log({f"{name} TNR": metrics["TNR"]}, step=step)
        wandb.log({f"{name} PPV": metrics["PPV"]}, step=step)
        wandb.log({f"{name} Theil_Index": metrics["Theil_Index"]}, step=step)
        # wandb.log_metric(f"{name} TPR, metrics['Theil_Index'])
        wandb.log({f"{name} Theil|s=1": metrics["Theil_Index_sex_Male_1.0"]}, step=step)
        wandb.log({f"{name} Theil_Index": metrics["Theil_Index"]}, step=step)
        wandb.log({f"{name} P(Y=1|s=0)": metrics["prob_pos_sex_Male_0.0"]}, step=step)
        wandb.log({f"{name} P(Y=1|s=1)": metrics["prob_pos_sex_Male_1.0"]}, step=step)
        wandb.log({f"{name} Theil|s=1": metrics["Theil_Index_sex_Male_1.0"]}, step=step)
        wandb.log({f"{name} Theil|s=0": metrics["Theil_Index_sex_Male_0.0"]}, step=step)
        wandb.log(
            {f"{name} P(Y=1|s=0) Ratio s0/s1": metrics["prob_pos_sex_Male_0.0/sex_Male_1.0"]},
            step=step,
        )
        wandb.log(
            {f"{name} P(Y=1|s=0) Diff s0-s1": metrics["prob_pos_sex_Male_0.0-sex_Male_1.0"]},
            step=step,
        )

        wandb.log({f"{name} TPR|s=1": metrics["TPR_sex_Male_1.0"]}, step=step)
        wandb.log({f"{name} TPR|s=0": metrics["TPR_sex_Male_0.0"]}, step=step)
        wandb.log({f"{name} TPR Ratio s0/s1": metrics["TPR_sex_Male_0.0/sex_Male_1.0"]}, step=step)
        wandb.log({f"{name} TPR Diff s0-s1": metrics["TPR_sex_Male_0.0/sex_Male_1.0"]}, step=step)

        wandb.log({f"{name} PPV Ratio s0/s1": metrics["PPV_sex_Male_0.0/sex_Male_1.0"]}, step=step)
        wandb.log({f"{name} TNR Ratio s0/s1": metrics["TNR_sex_Male_0.0/sex_Male_1.0"]}, step=step)
    else:
        metrics = run_metrics(predictions, actual, metrics=[Accuracy()], per_sens_metrics=[])
        wandb.log({f"{name} Accuracy": metrics["Accuracy"]}, step=step)
    return metrics


def fit_classifier(args, input_dim, train_data, train_on_recon, pred_s, test_data=None):

    if args.dataset == "cmnist":
        clf_fn = mp_32x32_net
    elif args.dataset == "celeba":
        clf_fn = mp_64x64_net
    else:
        clf_fn = fc_net
        input_dim = (input_dim,)
    clf = clf_fn(input_dim=input_dim, target_dim=args.y_dim)

    n_classes = args.y_dim if args.y_dim > 1 else 2
    clf: Classifier = Classifier(clf, num_classes=n_classes, optimizer_kwargs={"lr": args.eval_lr})
    clf.to(args.device)
    clf.fit(
        train_data, test_data=test_data, epochs=args.eval_epochs, device=args.device, pred_s=pred_s
    )

    return clf


def make_tuple_from_data(train, test, pred_s):
    train_x = train.x
    test_x = test.x

    if pred_s:
        train_y = train.s
        test_y = test.s
    else:
        train_y = train.y
        test_y = test.y

    return (DataTuple(x=train_x, s=train.s, y=train_y), DataTuple(x=test_x, s=test.s, y=test_y))


def evaluate(
    args,
    step,
    train_data,
    test_data,
    name,
    train_on_recon=True,
    pred_s=False,
    save_to_csv: Optional[Path] = None,
):
    input_dim = next(iter(train_data))[0].shape[0]

    if args.dataset == "cmnist":

        train_data = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
        test_data = DataLoader(
            test_data, batch_size=args.test_batch_size, shuffle=False, pin_memory=True
        )

        clf: Classifier = fit_classifier(
            args,
            input_dim,
            train_data=train_data,
            train_on_recon=train_on_recon,
            pred_s=pred_s,
            test_data=test_data,
        )

        preds, actual, sens = clf.predict_dataset(test_data, device=args.device)
        preds = pd.DataFrame(preds)
        actual = DataTuple(x=None, s=sens, y=pd.DataFrame(sens if pred_s else actual))

    else:
        if not isinstance(train_data, DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=pred_s)
        clf = LR()
        preds = clf.run(train_data, test_data)
        actual = test_data

    full_name = name
    full_name += "_s" if pred_s else "_y"
    full_name += "_on_recons" if train_on_recon else "_on_encodings"
    metrics = compute_metrics(preds, actual, full_name, run_all=args.dataset == "adult", step=step)
    print(f"Results for {full_name}:")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()  # empty line

    if save_to_csv is not None and args.results_csv:
        assert isinstance(save_to_csv, Path)
        results_path = save_to_csv / f"{full_name}_{args.results_csv}"
        value_list = ",".join([str(args.scale)] + [str(v) for v in metrics.values()])
        if results_path.is_file():
            with results_path.open("a") as f:
                f.write(value_list + "\n")
        else:
            with results_path.open("w") as f:
                f.write(",".join(["Scale"] + [str(k) for k in metrics.keys()]) + "\n")
                f.write(value_list + "\n")
        for name, value in metrics.items():
            wandb.run.summary[name] = value

    return metrics


def encode_dataset(
    args: Namespace,
    data: Dataset,
    model: BipartiteInn,
    recon: bool,
    subdir: str,
    get_zy: bool = False,
) -> dict:

    encodings = {"xy": []}
    if get_zy:
        encodings["zy"] = []
    all_s = []
    all_y = []

    data = DataLoader(data, batch_size=args.test_batch_size, pin_memory=True, shuffle=False)

    with torch.set_grad_enabled(False):
        for i, (x, s, y) in enumerate(data):

            x = x.to(args.device)
            all_s.append(s)
            all_y.append(y)

            z, zy, zs = model.encode(x, partials=True)

            zs_m = torch.cat([zy, torch.zeros_like(zs)], dim=1)
            xy = model.invert(zs_m)
            if x.dim() > 2:
                xy = xy.clamp(min=0, max=1)

            encodings["xy"].append(xy.detach().cpu())
            if get_zy:
                encodings["zy"].append(zy.detach().cpu())

    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encodings["xy"] = TensorDataset(torch.cat(encodings["xy"], dim=0), all_s, all_y)
    if get_zy:
        encodings["zy"] = TensorDataset(torch.cat(encodings["zy"], dim=0), all_s, all_y)

    return encodings

    # path = Path("data", "encodings", subdir)
    # if os.path.exists(path):
    #     shutil.rmtree(path)
    # os.mkdir(path)
    #
    # encodings = ['z', 'zy', 'zs']
    # if recon:
    #     encodings.extend(['x', 'xy', 'xs'])
    #
    # filepaths = {key: Path(path, key) for key in encodings}
    #
    # data = DataLoader(data, batch_size=args.test_batch_size, pin_memory=True, shuffle=False)
    #
    # index_offset = 0
    # with torch.set_grad_enabled(False):
    #     for i, (x, s, y) in enumerate(data):
    #         x = x.to(args.device)
    #
    #         z, zy, zs = model.encode(x, partials=True)
    #         if recon:
    #             x_recon, xy, xs = model.decode(z, partials=True)
    #
    #         for j in range(z.size(0)):
    #             file_index = index_offset + j
    #             s_j, y_j = s[j], y[j]
    #
    #             data_tuple_to_dataset_sample(z[j], s_j, y_j,
    #                                          root=filepaths['z'],
    #                                          filename=f"image_{file_index}")
    #
    #             data_tuple_to_dataset_sample(zy[j], s_j, y_j,
    #                                          root=filepaths['zy'],
    #                                          filename=f"image_{file_index}")
    #             data_tuple_to_dataset_sample(zs[j], s_j, y_j,
    #                                          root=filepaths['zs'],
    #                                          filename=f"image_{file_index}")
    #
    #             if recon:
    #                 data_tuple_to_dataset_sample(x_recon[j], s_j, y_j,
    #                                              root=filepaths['x'],
    #                                              filename=f"image_{file_index}")
    #                 data_tuple_to_dataset_sample(xy[j], s_j, y_j,
    #                                              root=filepaths['xy'],
    #                                              filename=f"image_{file_index}")
    #                 data_tuple_to_dataset_sample(xs[j], s_j, y_j,
    #                                              root=filepaths['xs'],
    #                                              filename=f"image_{file_index}")
    #
    #         index_offset += x.size(0)
    #
    # datasets = {
    #     key: TripletDataset(root)
    #     for key, root in filepaths.items()
    # }
    #
    # return datasets
