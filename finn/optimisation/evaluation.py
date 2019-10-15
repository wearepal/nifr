from pathlib import Path
from typing import Optional, Dict
import shutil
from argparse import Namespace
import torch.nn.functional as F
import pandas as pd
import torch
from ethicml.algorithms.inprocess import LR
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.metrics import Accuracy, Theil, ProbPos, TPR, TNR, PPV, NMI
from ethicml.utility.data_structures import DataTuple
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.utils import save_image

from finn.data import get_data_tuples
from finn.data.dataset_wrappers import TripletDataset
from finn.data.misc import data_tuple_to_dataset_sample
from finn.models.classifier import Classifier
from finn.models.configs import mp_28x28_net
from finn.models.configs.classifiers import fc_net, mp_32x32_net
from finn.models.inn import BipartiteInn


def compute_metrics(experiment, predictions, actual, name, run_all=False) -> Dict[str, float]:
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
        experiment.log_metric(f"{name} Accuracy", metrics["Accuracy"])
        experiment.log_metric(f"{name} TPR", metrics["TPR"])
        experiment.log_metric(f"{name} TNR", metrics["TNR"])
        experiment.log_metric(f"{name} PPV", metrics["PPV"])
        experiment.log_metric(f"{name} Theil_Index", metrics["Theil_Index"])
        # experiment.log_metric(f"{name} TPR, metrics['Theil_Index'])
        experiment.log_metric(f"{name} Theil|s=1", metrics["Theil_Index_sex_Male_1.0"])
        experiment.log_metric(f"{name} Theil_Index", metrics["Theil_Index"])
        experiment.log_metric(f"{name} P(Y=1|s=0)", metrics["prob_pos_sex_Male_0.0"])
        experiment.log_metric(f"{name} P(Y=1|s=1)", metrics["prob_pos_sex_Male_1.0"])
        experiment.log_metric(f"{name} Theil|s=1", metrics["Theil_Index_sex_Male_1.0"])
        experiment.log_metric(f"{name} Theil|s=0", metrics["Theil_Index_sex_Male_0.0"])
        experiment.log_metric(
            f"{name} P(Y=1|s=0) Ratio s0/s1", metrics["prob_pos_sex_Male_0.0/sex_Male_1.0"]
        )
        experiment.log_metric(
            f"{name} P(Y=1|s=0) Diff s0-s1", metrics["prob_pos_sex_Male_0.0-sex_Male_1.0"]
        )

        experiment.log_metric(f"{name} TPR|s=1", metrics["TPR_sex_Male_1.0"])
        experiment.log_metric(f"{name} TPR|s=0", metrics["TPR_sex_Male_0.0"])
        experiment.log_metric(f"{name} TPR Ratio s0/s1", metrics["TPR_sex_Male_0.0/sex_Male_1.0"])
        experiment.log_metric(f"{name} TPR Diff s0-s1", metrics["TPR_sex_Male_0.0/sex_Male_1.0"])

        experiment.log_metric(f"{name} PPV Ratio s0/s1", metrics["PPV_sex_Male_0.0/sex_Male_1.0"])
        experiment.log_metric(f"{name} TNR Ratio s0/s1", metrics["TNR_sex_Male_0.0/sex_Male_1.0"])
    else:
        metrics = run_metrics(predictions, actual, metrics=[Accuracy()], per_sens_metrics=[])
        experiment.log_metric(f"{name} Accuracy", metrics["Accuracy"])
    return metrics


def fit_classifier(args, input_dim, train_data, train_on_recon, pred_s, test_data=None):
    if train_on_recon or args.train_on_recon:
        clf = mp_32x32_net(input_dim=input_dim, target_dim=args.y_dim)
    else:
        clf = fc_net(input_dim, target_dim=args.y_dim)

    n_classes = args.y_dim if args.y_dim > 1 else 2
    clf: Classifier = Classifier(clf, num_classes=n_classes, optimizer_kwargs={"lr": args.eval_lr})
    clf.to(args.device)
    clf.fit(
        train_data,
        test_data=test_data,
        epochs=args.eval_epochs,
        device=args.device,
        pred_s=pred_s,
        verbose=True,
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
    experiment,
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
        actual = DataTuple(x=None, s=sens, y=pd.DataFrame(actual))

    else:
        if not isinstance(train_data, DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=pred_s)
        clf = LR()
        preds = clf.run(train_data, test_data)
        actual = test_data

    metrics = compute_metrics(experiment, preds, actual, name, run_all=args.dataset == "adult")
    res_type = "s" if pred_s else "y"
    res_type += "_on_recons" if train_on_recon else "_on_encodings"
    print(f"Results for '{name}', {res_type}:")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()  # empty line

    if save_to_csv is not None and args.results_csv:
        assert isinstance(save_to_csv, Path)
        res_type = "recon" if train_on_recon else "encoding"
        results_path = save_to_csv / f"{name}_{res_type}_{args.results_csv}"
        value_list = ",".join([str(args.scale)] + [str(v) for v in metrics.values()])
        if results_path.is_file():
            with results_path.open("a") as f:
                f.write(value_list + "\n")
        else:
            with results_path.open("w") as f:
                f.write(",".join(["Scale"] + [str(k) for k in metrics.keys()]) + "\n")
                f.write(value_list + "\n")

    return metrics


def encode_dataset(
    args: Namespace, data: Dataset, model: BipartiteInn, recon: bool, subdir: str
) -> dict:

    encodings = {"xy": []}
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

            encodings["xy"].append(xy.cpu())

    encodings["xy"] = TensorDataset(
        torch.cat(encodings["xy"], dim=0), torch.cat(all_s, dim=0), torch.cat(all_y, dim=0)
    )

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
