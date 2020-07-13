import random
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz
from matplotlib import cm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import wandb
from ethicml.algorithms.inprocess import LR
from ethicml.evaluators import run_metrics
from ethicml.metrics import NMI, PPV, TNR, TPR, Accuracy, ProbPos, RenyiCorrelation
from ethicml.utility import DataTuple, Prediction
from nosinn.configs import NosinnArgs, SharedArgs
from nosinn.data import DatasetTriplet, get_data_tuples
from nosinn.models import BipartiteInn, Classifier
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
    print("Encoding task dataset...")
    task_repr = encode_dataset(args, data.task, model, recon=True, subdir="task")
    print("Encoding task train dataset...")
    task_train_repr = encode_dataset(args, data.task_train, model, recon=True, subdir="task_train")

    print("\nComputing metrics...")
    _, clf = evaluate(
        args,
        step,
        task_train_repr["xy"],
        task_repr["xy"],
        name="xy",
        train_on_recon=True,
        pred_s=False,
        save_to_csv=save_to_csv,
    )

    if feat_attr and args.dataset != "adult":
        print("Creating feature attribution maps...")
        save_dir = Path(args.save_dir) / "feat_attr_maps"
        save_dir.mkdir(exist_ok=True, parents=True)  # create directory if it doesn't exist
        pred_orig, actual, _ = clf.predict_dataset(data.task, device=args.device)
        pred_deb, _, _ = clf.predict_dataset(task_repr["xy"], device=args.device)

        orig_correct = pred_orig == actual
        deb_correct = pred_deb == actual
        diff = (deb_correct.bool() & actual.bool()) ^ (deb_correct.bool() ^ orig_correct.bool())
        cand_inds = torch.arange(end=actual.size(0))[diff].tolist()

        num_samples = min(actual.size(0), 50)
        inds = random.sample(cand_inds, num_samples)

        if data.y_dim == 1:

            def _binary_clf_fn(self, _input):
                out = self.model(_input).sigmoid()
                return torch.cat([1 - out, out], dim=-1)

            clf.forward = types.MethodType(_binary_clf_fn, clf)

        clf.cpu()

        for k, _ in enumerate(inds):
            image_orig, _, target_orig = data.task[k]
            image_deb, _, target_deb = task_repr["xy"][k]

            if image_orig.dim() == 3:
                feat_attr_map_orig = get_image_attribution(image_orig, target_orig, clf)
                feat_attr_map_orig.savefig(save_dir / f"feat_attr_map_orig_{k}.png")

                feat_attr_map_deb = get_image_attribution(image_deb, target_deb, clf)
                feat_attr_map_deb.savefig(save_dir / f"feat_attr_map_deb{k}.png")

        clf.to(args.device)

    # print("===> Predict y from xy")
    # evaluate(args, experiment, repr.task_train['x'], repr.task['x'], name='xy', pred_s=False)
    # print("===> Predict s from xy")
    # evaluate(args, experiment, task_train_repr['xy'], task_repr['xy'], name='xy', pred_s=True)

    if quick_eval:
        log_sample_images(args, data.task_train, "task_train", step=step)
    else:

        if args.dataset == "adult":
            task_data, task_train_data = get_data_tuples(data.task, data.task_train)
            data = DatasetTriplet(
                pretrain=None,
                task=task_data,
                task_train=task_train_data,
                s_dim=data.s_dim,
                y_dim=data.y_dim,
            )

        # ===========================================================================

        evaluate(args, step, task_train_repr["zy"], task_repr["zy"], name="zy")
        evaluate(args, step, task_train_repr["zs"], task_repr["zs"], name="zs")
        evaluate(args, step, task_train_repr["xy"], task_repr["xy"], name="xy")
        evaluate(args, step, task_train_repr["xs"], task_repr["xs"], name="xs")


def compute_metrics(
    args: SharedArgs, predictions: Prediction, actual, name: str, step: int, run_all=False
) -> Dict[str, float]:
    """Compute accuracy and fairness metrics and log them"""

    if run_all:
        metrics = run_metrics(
            predictions,
            actual,
            metrics=[Accuracy(), ProbPos(), TPR(), TNR(), PPV(), RenyiCorrelation()],
            per_sens_metrics=[Accuracy(), ProbPos(), TPR(), TNR(), PPV()],
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
    train_on_recon: bool = True,
    pred_s: bool = False,
    save_to_csv: Optional[Path] = None,
):
    input_dim = next(iter(train_data))[0].shape[0]

    if args.dataset in ("cmnist", "celeba", "ssrp", "genfaces"):

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
    full_name += "_on_recons" if train_on_recon else "_on_encodings"
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
    model: BipartiteInn,
    recon: bool,
    subdir: str,
    get_zy: bool = False,
) -> Dict[str, torch.utils.data.Dataset]:

    encodings: Dict[str, List[torch.Tensor]] = {"xy": []}
    if get_zy:
        encodings["zy"] = []
    all_s = []
    all_y = []

    data = DataLoader(
        data, batch_size=args.encode_batch_size, pin_memory=True, shuffle=False, num_workers=4
    )

    with torch.set_grad_enabled(False):
        for _, (x, s, y) in enumerate(tqdm(data)):

            x = x.to(args.device, non_blocking=True)
            all_s.append(s)
            all_y.append(y)

            _, zy, zs = model.encode(x, partials=True)

            zs_m = torch.cat([zy, torch.zeros_like(zs)], dim=1)
            xy = model.invert(zs_m)

            if args.dataset in ("celeba", "ssrp", "genfaces"):
                xy = 0.5 * xy + 0.5

            if x.dim() > 2:
                xy = xy.clamp(min=0, max=1)

            encodings["xy"].append(xy.detach().cpu())
            if get_zy:
                encodings["zy"].append(zy.detach().cpu())

    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encodings_dt: Dict[str, torch.utils.data.Dataset] = {}
    encodings_dt["xy"] = TensorDataset(torch.cat(encodings["xy"], dim=0), all_s, all_y)
    if get_zy:
        encodings_dt["zy"] = TensorDataset(torch.cat(encodings["zy"], dim=0), all_s, all_y)

    return encodings_dt

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
