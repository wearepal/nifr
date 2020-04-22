from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
from matplotlib import cm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from typing_extensions import Literal

import wandb
from ethicml.algorithms.inprocess import LR
from ethicml.evaluators import run_metrics
from ethicml.metrics import NMI, PPV, TNR, TPR, Accuracy, ProbPos
from ethicml.utility import DataTuple, Prediction
from nosinn.configs import NosinnArgs, SharedArgs
from nosinn.data import DatasetTriplet, get_data_tuples
from nosinn.models import PartitionedAeInn, Classifier
from nosinn.models.configs import fc_net, mp_32x32_net, mp_64x64_net
from nosinn.utils import wandb_log

from .utils import log_images


def log_sample_images(args, data, name, step):
    data_loader = DataLoader(data, shuffle=False, batch_size=64)
    x, _, _ = next(iter(data_loader))
    log_images(args, x, f"Samples from {name}", prefix="eval", step=step)


def log_metrics(
    args: NosinnArgs,
    model,
    data: DatasetTriplet,
    step: int,
    quick_eval: bool = True,
    save_to_csv: Optional[Path] = None,
    check_originals: bool = False,
    feat_attr=False,
):
    """Compute and log a variety of metrics"""
    model.eval()
    print("Encoding training set...")
    train_inv_s = encode_dataset(
        args, data.train, model, recons=args.eval_on_recon, invariant_to="s"
    )
    if args.eval_on_recon:
        # don't encode test dataset
        test_repr = data.test
    else:
        test_repr = encode_dataset(args, data.test, model, recons=False, invariant_to="s")

    print("\nComputing metrics...")
    evaluate(
        args,
        step,
        train_inv_s,
        test_repr,
        name="x_rand_s",
        eval_on_recon=args.eval_on_recon,
        pred_s=False,
        save_to_csv=save_to_csv,
    )

    print("Encoding training dataset (random y)...")
    train_rand_y = encode_dataset(
        args, data.train, model, recons=args.eval_on_recon, invariant_to="y"
    )
    evaluate(
        args,
        step,
        train_rand_y,
        test_repr,
        name="x_rand_y",
        eval_on_recon=args.eval_on_recon,
        pred_s=False,
        save_to_csv=save_to_csv,
    )


def compute_metrics(
    args: SharedArgs, predictions: Prediction, actual, name: str, step: int, run_all=False
) -> Dict[str, float]:
    """Compute accuracy and fairness metrics and log them"""

    if run_all:
        metrics = run_metrics(
            predictions,
            actual,
            metrics=[Accuracy(), TPR(), TNR(), PPV(), NMI(base="y"), NMI(base="s")],
            per_sens_metrics=[ProbPos(), TPR(), TNR(), PPV(), NMI(base="y"), NMI(base="s")],
        )
        logging_dict = {
            f"{name} Accuracy": metrics["Accuracy"],
            f"{name} TPR": metrics["TPR"],
            f"{name} TNR": metrics["TNR"],
            f"{name} PPV": metrics["PPV"],
            f"{name} P(Y=1|s=0)": metrics["prob_pos_sex_Male_0.0"],
            f"{name} P(Y=1|s=1)": metrics["prob_pos_sex_Male_1.0"],
            f"{name} P(Y=1|s=0) Ratio s0/s1": metrics["prob_pos_sex_Male_0.0/sex_Male_1.0"],
            f"{name} P(Y=1|s=0) Diff s0-s1": metrics["prob_pos_sex_Male_0.0-sex_Male_1.0"],
            f"{name} TPR|s=1": metrics["TPR_sex_Male_1.0"],
            f"{name} TPR|s=0": metrics["TPR_sex_Male_0.0"],
            f"{name} TPR Ratio s0/s1": metrics["TPR_sex_Male_0.0/sex_Male_1.0"],
            f"{name} TPR Diff s0-s1": metrics["TPR_sex_Male_0.0/sex_Male_1.0"],
            f"{name} PPV Ratio s0/s1": metrics["PPV_sex_Male_0.0/sex_Male_1.0"],
            f"{name} TNR Ratio s0/s1": metrics["TNR_sex_Male_0.0/sex_Male_1.0"],
        }
        wandb_log(args, logging_dict, step=step)
    else:
        metrics = run_metrics(predictions, actual, metrics=[Accuracy()], per_sens_metrics=[])
        wandb_log(args, {f"{name} Accuracy": metrics["Accuracy"]}, step=step)
    return metrics


def fit_classifier(args, input_dim, train_data, train_on_recon, pred_s, test_data=None):

    if args.dataset == "cmnist":
        clf_fn = mp_32x32_net
    elif args.dataset in ("celeba", "ssrp", "genfaces"):
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


def get_image_attribution(input, target, model):

    if input.dim() == 3:
        original_image = np.transpose(input.cpu().detach().numpy(), (1, 2, 0))
        input = input.unsqueeze(0)
    else:
        original_image = np.transpose(input.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input, target=target, **kwargs)

        return tensor_attributions

    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)

    attr_ig_nt = attribute_image_features(
        nt, input, baselines=input * 0, nt_type="smoothgrad_sq", n_samples=100, stdevs=0.2
    )
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    cmap = cm.get_cmap("viridis", 12)
    fig, ax = viz.visualize_image_attr_multiple(
        attr_ig_nt,
        original_image=original_image,
        methods=["original_image", "masked_image", "blended_heat_map"],
        signs=[None, "absolute_value", "absolute_value"],
        outlier_perc=10,
        cmap=cmap,
        show_colorbar=True,
        use_pyplot=False,
    )

    return fig


def evaluate(
    args: SharedArgs,
    step: int,
    train_data,
    test_data,
    name: str,
    eval_on_recon: bool = True,
    pred_s: bool = False,
    save_to_csv: Optional[Path] = None,
):
    input_shape = next(iter(train_data))[0].shape

    if args.dataset in ("cmnist", "celeba", "ssrp", "genfaces"):

        train_data = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
        test_data = DataLoader(
            test_data, batch_size=args.test_batch_size, shuffle=False, pin_memory=True
        )

        clf: Classifier = fit_classifier(
            args,
            input_shape,
            train_data=train_data,
            train_on_recon=eval_on_recon,
            pred_s=pred_s,
            test_data=test_data,
        )

        preds, actual, sens = clf.predict_dataset(test_data, device=args.device)
        preds = Prediction(hard=pd.Series(preds))
        sens_pd = pd.DataFrame(sens.numpy().astype(np.float32), columns=["sex_Male"])
        labels = pd.DataFrame(actual, columns=["labels"])
        actual = DataTuple(x=sens_pd, s=sens_pd, y=sens_pd if pred_s else labels)

    else:
        if not isinstance(train_data, DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(train_data, test_data, pred_s=pred_s)
        clf = LR()
        preds = clf.run(train_data, test_data)
        actual = test_data

    full_name = f"{args.dataset}_{name}"
    full_name += "_s" if pred_s else "_y"
    full_name += "_on_recons" if eval_on_recon else "_on_encodings"
    metrics = compute_metrics(args, preds, actual, full_name, run_all=args.y_dim == 1, step=step)
    print(f"Results for {full_name}:")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()  # empty line

    if save_to_csv is not None and args.results_csv:
        assert isinstance(save_to_csv, Path)
        sweep_key = "Scale" if args.dataset == "cmnist" else "Mix_fact"
        sweep_value = str(args.scale) if args.dataset == "cmnist" else str(args.task_mixing_factor)
        results_path = save_to_csv / f"{full_name}_{args.results_csv}"
        value_list = ",".join([sweep_value] + [str(v) for v in metrics.values()])
        if not results_path.is_file():
            with results_path.open("w") as f:
                f.write(",".join([sweep_key] + [str(k) for k in metrics.keys()]) + "\n")  # header
                f.write(value_list + "\n")
        else:
            with results_path.open("a") as f:  # append to existing file
                f.write(value_list + "\n")
        print(f"Results have been written to {results_path.resolve()}")
        if args.use_wandb:
            for metric_name, value in metrics.items():
                wandb.run.summary[metric_name] = value

    return metrics, clf


def encode_dataset(
    args: SharedArgs,
    data: Dataset,
    model: PartitionedAeInn,
    recons: bool,
    invariant_to: Literal["s", "y"] = "s",
) -> Dict[str, torch.utils.data.Dataset]:

    all_x_m = []
    all_s = []
    all_y = []

    data_loader = DataLoader(
        data, batch_size=args.encode_batch_size, pin_memory=True, shuffle=False, num_workers=4
    )

    with torch.set_grad_enabled(False):
        for x, s, y in tqdm(data_loader):

            x = x.to(args.device, non_blocking=True)
            all_s.append(s)
            all_y.append(y)

            enc = model.encode(x)
            if recons:
                zs_m, zy_m = model.mask(enc, random=True)
                z_m = zs_m if invariant_to == "s" else zy_m
                x_m = model.decode(z_m, discretize=True)

                if args.dataset in ("celeba", "ssrp", "genfaces"):
                    x_m = 0.5 * x_m + 0.5
                if x.dim() > 2:
                    x_m = x_m.clamp(min=0, max=1)
            else:
                zs_m, zy_m = model.mask(enc)
                # `zs_m` has zs zeroed out
                x_m = zs_m if invariant_to == "s" else zy_m

            all_x_m.append(x_m.detach().cpu())

    all_x_m = torch.cat(all_x_m, dim=0)
    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encoded_dataset = TensorDataset(all_x_m, all_s, all_y)
    print("Done.")

    return encoded_dataset

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
