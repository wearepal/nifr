"""Baseline for Adult dataset"""
import pandas as pd

from ethicml.algorithms.inprocess import LR, SVM, Agarwal, Kamiran, Majority
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.metrics import CV, NMI, PPV, TNR, TPR, Accuracy, ProbPos, Theil
from ethicml.preprocessing.train_test_split import train_test_split
from nifr.data.data_loading import load_adult_data_tuples


def main():
    class _Namespace:
        pretrain = True
        drop_native = True
        meta_lead = True
        data_split_seed = 888

    for clf in [LR()]:  # , Majority(), Kamiran(), Agarwal(), SVM(kernel='linear')]:
        df = pd.DataFrame(
            columns=[
                "mix_factor",
                "Accuracy",
                "Theil_Index",
                "TPR_sex_Male_1",
                "TPR_sex_Male_0",
                "TPR_sex_Male_0-sex_Male_1",
                "TPR_sex_Male_0/sex_Male_1",
                "prob_pos_sex_Male_1",
                "prob_pos_sex_Male_0",
                "prob_pos_sex_Male_0-sex_Male_1",
                "prob_pos_sex_Male_0/sex_Male_1",
                "Theil_Index_sex_Male_1",
                "Theil_Index_sex_Male_0",
                "Theil_Index_sex_Male_0-sex_Male_1",
                "Theil_Index_sex_Male_0/sex_Male_1",
                "NMI",
                "NMI_sex_Male_0",
                "NMI_sex_Male_0-sex_Male_1",
                "NMI_sex_Male_0/sex_Male_1",
                "NMI_sex_Male_1",
                "PPV",
                "PPV_sex_Male_0",
                "PPV_sex_Male_0-sex_Male_1",
                "PPV_sex_Male_0/sex_Male_1",
                "PPV_sex_Male_1",
                "TNR",
                "TNR_sex_Male_0",
                "TNR_sex_Male_0-sex_Male_1",
                "TNR_sex_Male_0/sex_Male_1",
                "TNR_sex_Male_1",
                "TPR",
            ]
        )
        for mix_fact in [k / 100 for k in range(0, 105, 5)]:
            args = _Namespace()
            args.task_mixing_factor = mix_fact
            _, task, task_train = load_adult_data_tuples(args)
            try:
                preds = clf.run(task_train, task)
            except:
                print(f"{clf.name} failed on mix: {mix_fact}")
                continue
            metrics = run_metrics(preds, task, metrics=[Accuracy()], per_sens_metrics=[])
            print(f"{clf.name} Accuracy: {metrics['Accuracy']}")
            res_dict = run_metrics(
                preds,
                task,
                metrics=[Accuracy(), Theil(), NMI(), TPR(), TNR(), PPV()],
                per_sens_metrics=[Theil(), ProbPos(), TPR(), TNR(), NMI(), PPV()],
            )
            res_dict["mix_factor"] = mix_fact

            print(",".join(res_dict.keys()))
            print(",".join([str(val) for val in res_dict.values()]))

            df = df.append(res_dict, ignore_index=True)
            print(f"mix: {mix_fact}\t acc: {res_dict['Accuracy']}")
            print("----------------------")
        df.to_csv(f"{clf.name}.csv", index=False)


if __name__ == "__main__":
    main()
