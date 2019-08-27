import os
import os
import shutil

import pandas as pd
import torch
from ethicml.algorithms.inprocess import LR, MLP
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.metrics import Accuracy, Theil, ProbPos, TPR, TNR, PPV, NMI

# from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
from ethicml.utility.data_structures import DataTuple
from torch.utils.data import DataLoader

from finn.data import DatasetTuple, get_data_tuples
from finn.data.datasets import TripletDataset
from finn.data.misc import data_tuple_to_dataset
from finn.models.classifier import Classifier
from finn.models.configs import mp_28x28_net
from finn.models.configs.classifiers import fc_net


def compute_metrics(experiment, predictions, actual, name, run_all=False):
    """Compute accuracy and fairness metrics and log them"""

    if run_all:
        metrics = run_metrics(
            predictions,
            actual,
            metrics=[Accuracy(), Theil(), TPR(), TNR(), PPV(), NMI(base='y'), NMI(base='s')],
            per_sens_metrics=[
                Theil(),
                ProbPos(),
                TPR(),
                TNR(),
                PPV(),
                NMI(base='y'),
                NMI(base='s'),
            ],
        )
        experiment.log_metric(f"{name} Accuracy", metrics['Accuracy'])
        experiment.log_metric(f"{name} TPR", metrics['TPR'])
        experiment.log_metric(f"{name} TNR", metrics['TNR'])
        experiment.log_metric(f"{name} PPV", metrics['PPV'])
        experiment.log_metric(f"{name} Theil_Index", metrics['Theil_Index'])
        # experiment.log_metric(f"{name} TPR, metrics['Theil_Index'])
        experiment.log_metric(f"{name} Theil|s=1", metrics['Theil_Index_sex_Male_1.0'])
        experiment.log_metric(f"{name} Theil_Index", metrics['Theil_Index'])
        experiment.log_metric(f"{name} P(Y=1|s=0)", metrics['prob_pos_sex_Male_0.0'])
        experiment.log_metric(f"{name} P(Y=1|s=1)", metrics['prob_pos_sex_Male_1.0'])
        experiment.log_metric(f"{name} Theil|s=1", metrics['Theil_Index_sex_Male_1.0'])
        experiment.log_metric(f"{name} Theil|s=0", metrics['Theil_Index_sex_Male_0.0'])
        experiment.log_metric(
            f"{name} P(Y=1|s=0) Ratio s0/s1", metrics['prob_pos_sex_Male_0.0/sex_Male_1.0']
        )
        experiment.log_metric(
            f"{name} P(Y=1|s=0) Diff s0-s1", metrics['prob_pos_sex_Male_0.0-sex_Male_1.0']
        )

        experiment.log_metric(f"{name} TPR|s=1", metrics['TPR_sex_Male_1.0'])
        experiment.log_metric(f"{name} TPR|s=0", metrics['TPR_sex_Male_0.0'])
        experiment.log_metric(f"{name} TPR Ratio s0/s1", metrics['TPR_sex_Male_0.0/sex_Male_1.0'])
        experiment.log_metric(f"{name} TPR Diff s0-s1", metrics['TPR_sex_Male_0.0/sex_Male_1.0'])

        experiment.log_metric(f"{name} PPV Ratio s0/s1", metrics['PPV_sex_Male_0.0/sex_Male_1.0'])
        experiment.log_metric(f"{name} TNR Ratio s0/s1", metrics['TNR_sex_Male_0.0/sex_Male_1.0'])
    else:
        metrics = run_metrics(predictions, actual, metrics=[Accuracy()], per_sens_metrics=[])
        experiment.log_metric(f"{name} Accuracy", metrics['Accuracy'])
    for key, value in metrics.items():
        print(f"\t\t{key}: {value:.4f}")
    print()  # empty line
    return metrics


def fit_classifier(args, input_dim, train_data, train_on_recon,
                   pred_s, test_data=None):
    if train_on_recon or args.learn_mask:
        clf = mp_28x28_net(input_dim=input_dim, target_dim=args.y_dim)
    else:
        clf = fc_net(input_dim, target_dim=args.y_dim)

    n_classes = args.y_dim if args.y_dim > 1 else 2
    clf: Classifier = Classifier(clf, n_classes=n_classes,
                                 optimizer_args={'lr': args.eval_lr})
    clf.to(args.device)
    clf.fit(train_data, test_data=test_data, epochs=args.eval_epochs,
            device=args.device, pred_s=pred_s, verbose=False)

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

    return DataTuple(x=train_x, s=train.s, y=train_y), DataTuple(x=test_x, s=test.s, y=test_y)


def evaluate(args, experiment, train_data, test_data,
             name, train_on_recon=True, pred_s=False):
    input_dim = next(iter(train_data))[0].shape[0]

    if args.dataset == 'cmnist':

        train_data = DataLoader(train_data, batch_size=args.batch_size,
                                shuffle=True, pin_memory=True)
        test_data = DataLoader(test_data, batch_size=args.test_batch_size,
                               shuffle=False, pin_memory=True)

        clf: Classifier = fit_classifier(args,
                                         input_dim,
                                         train_data=train_data,
                                         train_on_recon=train_on_recon,
                                         pred_s=pred_s,
                                         test_data=test_data)

        preds, actual, sens = clf.predict_dataset(test_data, device=args.device)
        preds = pd.DataFrame(preds)
        actual = DataTuple(x=None, s=sens, y=pd.DataFrame(actual))

    else:
        if not isinstance(train_data, DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)

        train_data, test_data = make_tuple_from_data(
            train_data, test_data, pred_s=pred_s,
        )
        clf = LR()
        preds = clf.run(train_data, test_data)
        actual = test_data

    print("\nComputing metrics...")
    _ = compute_metrics(experiment, preds, actual, name, run_all=args.dataset == 'adult')


def encode_dataset(args, data, model, recon):
    root = os.path.join('data', 'encodings')
    if os.path.exists(root):
        shutil.rmtree(root)
    os.mkdir(root)

    encodings = ['z', 'zy', 'zs']
    if recon:
        encodings.extend(['x_recon', 'xy', 'xs'])

    filepaths = {key: os.path.join(root, key) for key in encodings}

    data = DataLoader(data, batch_size=1, pin_memory=True)

    with torch.set_grad_enabled(False):
        for i, (x, s, y) in enumerate(iter(data)):
            x = x.to(args.device)
            s = s.to(args.device)

            z, zy, zs = model.encode(x, partials=True)

            data_tuple_to_dataset(z, s, y,
                                  root=filepaths['z'],
                                  filename=f"image_{i}")
            data_tuple_to_dataset(zy, s, y,
                                  root=filepaths['zy'],
                                  filename=f"image_{i}")
            data_tuple_to_dataset(zs, s, y,
                                  root=os.path.join(root, 'zs'),
                                  filename=f"image_{i}")

            if recon:
                x_recon, xy, xs = model.decode(z, partials=True)

                data_tuple_to_dataset(x_recon, s, y,
                                      root=filepaths['x_recon'],
                                      filename=f"image_{i}")
                data_tuple_to_dataset(xy, s, y,
                                      root=filepaths['xy'],
                                      filename=f"image_{i}")
                data_tuple_to_dataset(xs, s, y,
                                      root=filepaths['xs'],
                                      filename=f"image_{i}")

    datasets = {
        key: TripletDataset(root)
        for key, root in filepaths.items()
    }

    return datasets
