import os

from ethicml.algorithms.inprocess import LR, MLP
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.metrics import Accuracy, Theil, ProbPos, TPR, TNR, PPV, NMI

# from ethicml.algorithms.preprocess.threaded.threaded_pre_algorithm import BasicTPA
from ethicml.utility.data_structures import DataTuple

from finn.utils.eval_metrics import evaluate_with_classifier
from finn.optimisation.training_utils import train_and_evaluate_classifier
from finn.data import MetaDataset, get_data_tuples


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


def metrics_for_pretrain(args, experiment, clf, repr, data, save_to_csv=False):
    assert isinstance(repr, MetaDataset)
    print('Meta Learn Results...')
    if args.dataset == 'cmnist':
        acc = evaluate_with_classifier(args, repr.task_train['zy'], repr.task['zy'], args.zy_dim)
        experiment.log_metric("Meta Accuracy", acc)
        print(f"Meta Accuracy: {acc:.4f}")
    else:
        if not isinstance(repr.task_train['zy'], DataTuple):
            repr.task_train['zy'], repr.task['zy'] = get_data_tuples(
                repr.task_train['zy'], repr.task['zy']
            )
        preds_meta = clf.run(repr.task_train['zy'], repr.task['zy'])
        metrics = compute_metrics(experiment, preds_meta, repr.task['zy'], "Meta", run_all=True)
        print(",".join(metrics.keys()))
        print(",".join([str(val) for val in metrics.values()]))

        if save_to_csv:
            if os.path.isfile('./adult_results.csv'):
                with open('./adult_results.csv', 'a') as f:
                    f.write("%s," % args.task_mixing_factor)
                    for key in metrics.keys():
                        f.write("%s," % metrics[key])
                    f.write("\n")
            else:
                with open('./adult_results.csv', 'w') as f:
                    f.write("Mix_fact,")
                    for key in metrics.keys():
                        f.write("%s," % key)
                    f.write("\n")
                    f.write("%s," % args.task_mixing_factor)
                    for key in metrics.keys():
                        f.write("%s," % metrics[key])
                    f.write("\n")


def make_tuple_from_data(train, test, pred_s, use_s):
    if use_s:
        raise RuntimeError("This shouldn't be reached.")
        # train_x = pd.concat([train.x, train.s], axis='columns')
        # test_x = pd.concat([test.x, test.s], axis='columns')
    else:
        train_x = train.x
        test_x = test.x

    if pred_s:
        train_y = train.s
        test_y = test.s
    else:
        train_y = train.y
        test_y = test.y

    return DataTuple(x=train_x, s=train.s, y=train_y), DataTuple(x=test_x, s=test.s, y=test_y)


def get_name(use_s, use_x, predict_y, use_fair, use_unfair):
    name = ""

    if predict_y:
        name += "pred y "
    else:
        name += "pred s "

    if use_x:
        if use_fair:
            name += "Recon Z not S "
        elif use_unfair:
            name += "Recon Z S "
        else:
            name += "Original x "
    else:
        if use_fair and use_unfair:
            name += "all Z "
        elif use_fair:
            name += "Fair Z "
        elif use_unfair:
            name += "Unfair Z "
        else:
            raise NotImplementedError("Not sure how this was reached")
    if use_s:
        name += "& s "
    else:
        name += ""

    return name


def evaluate_representations(
    args,
    experiment,
    train_data,
    test_data,
    predict_y=False,
    use_s=False,
    use_x=False,
    use_fair=False,
    use_unfair=False,
):

    name = get_name(use_s, use_x, predict_y, use_fair, use_unfair)

    if args.pretrain:
        if use_fair and use_unfair:
            in_channels = args.zy_dim + args.zs_dim
        elif use_fair:
            in_channels = args.zy_dim
        else:
            in_channels = args.zs_dim
    else:
        if use_fair and use_unfair:
            in_channels = args.zy_dim + args.zs_dim + args.zn_dim
        elif use_fair:
            in_channels = args.zy_dim
        elif use_unfair:
            in_channels = args.zs_dim
        else:
            in_channels = args.zn_dim

    print(f"{name}:")

    if args.dataset == 'cmnist':
        run_all = False
        if use_x:
            clf = train_and_evaluate_classifier(
                args, experiment, train_data, pred_s=not predict_y, use_s=use_s, name=name
            )
        else:
            clf = evaluate_with_classifier(
                args,
                train_data,
                test_data,
                in_channels=in_channels,
                pred_s=not predict_y,
                use_s=use_s,
                applicative=True,
            )
        preds_x, test_x = clf(test_data=train_data)
        _ = compute_metrics(experiment, preds_x, test_x, f"{name} - Train")
        preds_x, test_x = clf(test_data=test_data)
        print("\tTraining performance")

    else:
        if not isinstance(train_data, DataTuple):
            train_data, test_data = get_data_tuples(train_data, test_data)
        run_all = True
        train_x, test_x = make_tuple_from_data(
            train_data, test_data, pred_s=not predict_y, use_s=use_s
        )
        clf = MLP() if args.mlp_clf else LR()
        preds_x = clf.run(train_x, test_x)

    print("\tTest performance")
    _ = compute_metrics(experiment, preds_x, test_x, name, run_all=run_all)
