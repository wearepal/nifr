"""Baseline for Adult dataset"""
import pandas as pd

from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.algorithms.inprocess import LR, SVM, Majority, Kamiran, Agarwal
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.metrics import Accuracy, CV, Theil, TPR, ProbPos
from ethicml.algorithms.utils import apply_to_joined_tuple, DataTuple

from finn.data.dataloading import load_adult_data


def main():
    class _Namespace:
        meta_learn = True

    args = _Namespace()
    args.drop_native = True
    args.task_mixing_factor = 0
    args.meta_lead = True
    meta_train, task, task_train = load_adult_data(args)
    ethicml_model = SVM(kernel='linear')
    # ethicml_model = Majority()
    predictions = ethicml_model.run(task_train, task)
    metrics = run_metrics(predictions, task, metrics=[Accuracy()], per_sens_metrics=[])
    print(f"Accuracy: {metrics['Accuracy']}")

    for clf in [Agarwal(), LR(), Majority(), Kamiran()]:
        df = pd.DataFrame(columns=[
            'mix_factor', 'Accuracy', 'CV', 'Theil_Index', 'Accuracy_sex_Male_1', 'Accuracy_sex_Male_0',
            'Accuracy_sex_Male_0-sex_Male_1', 'Accuracy_sex_Male_0/sex_Male_1', 'TPR_sex_Male_1', 'TPR_sex_Male_0',
            'TPR_sex_Male_0-sex_Male_1', 'TPR_sex_Male_0/sex_Male_1', 'prob_pos_sex_Male_1', 'prob_pos_sex_Male_0',
            'prob_pos_sex_Male_0-sex_Male_1', 'prob_pos_sex_Male_0/sex_Male_1', 'Theil_Index_sex_Male_1',
            'Theil_Index_sex_Male_0', 'Theil_Index_sex_Male_0-sex_Male_1', 'Theil_Index_sex_Male_0/sex_Male_1'
        ])
        for mix_fact in [k / 100 for k in range(0, 105, 5)]:
            args = _Namespace()
            args.drop_native = True
            args.task_mixing_factor = mix_fact
            args.meta_lead = True
            meta_train, task, task_train = load_adult_data(args)
            try:
                preds = clf.run(task_train, task)
            except:
                print(f"{clf.name} failed on mix: {mix_fact}")
                continue
            metrics = run_metrics(preds, task, metrics=[Accuracy()], per_sens_metrics=[])
            print(f"{clf.name} Accuracy: {metrics['Accuracy']}")
            res_dict = run_metrics(preds, task, [Accuracy(), CV(), Theil()], [Accuracy(), TPR(), ProbPos(), Theil()])
            res_dict['mix_factor'] = mix_fact
            df = df.append(res_dict, ignore_index=True)
            print(f"mix: {mix_fact}\t acc: {res_dict['Accuracy']}")
            print("----------------------")
        df.to_csv(f'{clf.name}.csv', index=False)


if __name__ == "__main__":
    main()
