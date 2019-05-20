"""Run the model and evaluate the fairness"""
# import sys
# from pathlib import Path
import comet_ml  # this import is needed because comet_ml has to be imported before sklearn

import random
import numpy as np
import torch

from ethicml.algorithms.inprocess.logistic_regression import LR
# from ethicml.algorithms.inprocess.svm import SVM
from torch.utils.data import DataLoader

from finn.train import main as training_loop
from finn.data import MetaDataset, get_data_tuples, load_dataset
from finn.utils.evaluate_utils import metrics_for_meta_learn, evaluate_representations
from finn.utils.training_utils import parse_arguments, encode_dataset, encode_dataset_no_recon, log_images
from finn.utils.eval_metrics import train_zy_head


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def main(raw_args=None):
    args = parse_arguments(raw_args)
    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)
    datasets = load_dataset(args)
    training_loop(args, datasets, log_metrics)

def log_sample_images(experiment, data, name):
    data_loader = DataLoader(data, shuffle=False, batch_size=65)
    x, s, y = next(iter(data_loader))
    log_images(experiment, x, f"Samples from {name}", prefix='eval')

def log_metrics(args, experiment, model, discs, data, check_originals=False):
    """Compute and log a variety of metrics"""
    assert args.meta_learn
    ethicml_model = LR()

    # This should be done before computing the representations because computing the representations
    # takes very long and is not needed for this!
    if args.meta_learn and args.inv_disc:
        assert isinstance(data, MetaDataset)
        acc = train_zy_head(args, experiment, model, discs, data.task_train, data.task)
        experiment.log_metric("Meta Accuracy", acc)
        print(f"Meta Accuracy: {acc:.4f}")
        # return
        model = discs.assemble_whole_model(model).eval()
    if check_originals:
        print("Evaluating on original dataset...")
        evaluate_representations(args, experiment, data.task_train, data.task,
                                 predict_y=True, use_x=True)
        evaluate_representations(args, experiment, data.task_train, data.task,
                                 predict_y=True, use_x=True, use_s=True)

    quick_eval = True
    if args.meta_learn and quick_eval:
        print("Quickly encode task and task train...")
        task_repr = encode_dataset_no_recon(args, data.task, model, recon_zyn=True)
        task_train_repr = encode_dataset_no_recon(args, data.task_train, model, recon_zyn=True)
        log_sample_images(experiment, data.task_train, "task_train")
        evaluate_representations(args, experiment, task_train_repr['recon_yn'], task_repr['recon_yn'],
                                 predict_y=True, use_x=True, use_fair=True, use_s=True)
        if args.dataset == 'adult':
            repr_ = MetaDataset(meta_train=None, task=task_repr, task_train=task_train_repr,
                                input_dim=data.input_dim, output_dim=data.output_dim)
            metrics_for_meta_learn(args, experiment, ethicml_model, repr_, data)
        return

    print('Encoding task dataset...')
    task_repr_ = encode_dataset(args, data.task, model)
    print('Encoding task train dataset...')
    task_train_repr_ = encode_dataset(args, data.task_train, model)

    repr = MetaDataset(meta_train=None, task=task_repr_, task_train=task_train_repr_,
                       input_dim=data.input_dim, output_dim=data.output_dim)

    if args.meta_learn and not args.inv_disc:
        metrics_for_meta_learn(args, experiment, ethicml_model, repr, data)

    if args.dataset == 'adult':
        task_data, task_train_data = get_data_tuples(data.task, data.task_train)
        data = MetaDataset(meta_train=None, task=task_data, task_train=task_train_data,
                           input_dim=data.input_dim, output_dim=data.output_dim)

    experiment.log_other("evaluation model", ethicml_model.name)

    # ===========================================================================
    if check_originals:
        evaluate_representations(args, experiment, data.task_train, data.task,
                                 predict_y=True, use_x=True)
        evaluate_representations(args, experiment, data.task_train, data.task,
                                 predict_y=True, use_x=True, use_s=True)

    # ===========================================================================

    evaluate_representations(args, experiment, repr.task_train['all_z'], repr.task['all_z'],
                             predict_y=True, use_fair=True, use_unfair=True)
    evaluate_representations(args, experiment, repr.task_train['recon_y'], repr.task['recon_y'],
                             predict_y=True, use_x=True, use_fair=True, use_s=True)
    evaluate_representations(args, experiment, repr.task_train['recon_s'], repr.task['recon_s'],
                             predict_y=True, use_x=True, use_unfair=True, use_s=True)

    check_grayscale = False
    if check_grayscale:
        # Grayscale the fair representation
        evaluate_representations(args, experiment, repr.task_train['recon_y'], repr.task['recon_y'],
                                 predict_y=True, use_x=True, use_fair=True)

    # ===========================================================================
    if check_originals:
        evaluate_representations(args, experiment, data.task_train, data.task, use_x=True)
        evaluate_representations(args, experiment, data.task_train, data.task, use_s=True, use_x=True)

    # ===========================================================================
    check_meta_train = False
    if check_meta_train:
        print('Encoding training set...')
        meta_train_repr = encode_dataset(args, data.meta_train, model)
        evaluate_representations(args, experiment, meta_train_repr['zy'], repr.task['zy'], use_fair=True)
        evaluate_representations(args, experiment, meta_train_repr['zs'], repr.task['zs'], use_unfair=True)


if __name__ == "__main__":
    main()
