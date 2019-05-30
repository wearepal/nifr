import os
from pathlib import Path
from typing import NamedTuple

from typing import Tuple
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from ethicml.algorithms.utils import DataTuple

from .cmnist import CMNIST
from .colorized_mnist import ColorizedMNIST
from .preprocess_cmnist import get_path_from_args
from .adult import load_adult_data


class MetaDataset(NamedTuple):
    meta_train: Dataset
    task: Dataset
    task_train: Dataset
    inner_meta: Tuple[Dataset, Dataset]
    input_dim: int
    output_dim: int


def load_dataset(args):
    assert args.meta_learn

    # =============== get whole dataset ===================
    if args.dataset == 'cmnist':
        cmnist_transforms = []
        if args.rotate_data:
            cmnist_transforms.append(transforms.RandomAffine(degrees=15))
        if args.shift_data:
            cmnist_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)))

        cmnist_transforms.append(transforms.ToTensor())
        cmnist_transforms = transforms.Compose(cmnist_transforms)

        whole_train_data = ColorizedMNIST(args.root, train=True,
                                          download=True, transform=cmnist_transforms,
                                          scale=args.scale,
                                          cspace=args.cspace,
                                          background=args.background,
                                          black=args.black,
                                          binarize=args.binarize)
        whole_test_data = ColorizedMNIST(args.root, train=False,
                                         download=True, transform=cmnist_transforms,
                                         scale=args.scale,
                                         cspace=args.cspace,
                                         background=args.background,
                                         black=args.black,
                                         binarize=args.binarize)

        # train_data, test_data = load_cmnist_from_file(args)
        args.y_dim = 10
        args.s_dim = 10

        whole_train_data.swap_train_test_colorization()
        whole_test_data.swap_train_test_colorization()
        # split the training set to get training and validation sets
        whole_train_data, whole_val_data = random_split(whole_train_data, lengths=(50000, 10000))

    else:
        meta_tuple, task_tuple, task_train_tuple = load_adult_data(args)
        whole_train_data = TensorDataset(*[torch.tensor(df.values, dtype=torch.float32)
                                           for df in meta_tuple])
        whole_val_data = TensorDataset(*[torch.tensor(df.values, dtype=torch.float32)
                                           for df in task_tuple])
        whole_test_data = TensorDataset(*[torch.tensor(df.values, dtype=torch.float32)
                                          for df in task_train_tuple])
        args.y_dim = 1
        args.s_dim = 1

    # shrink meta train set according to args.data_pcnt
    meta_train_len = int(args.data_pcnt * len(whole_train_data))
    meta_train_data, _ = random_split(
        whole_train_data, lengths=(meta_train_len, len(whole_train_data) - meta_train_len))

    # shrink task set according to args.data_pcnt
    task_len = int(args.data_pcnt * len(whole_val_data))
    task_data, _ = random_split(whole_val_data, lengths=(task_len, len(whole_val_data) - task_len))
    task_data.transform = transforms.ToTensor()
    # shrink task train set according to args.data_pcnt
    task_train_len = int(args.data_pcnt * len(whole_test_data))
    task_train_data, _ = random_split(
        whole_test_data, lengths=(task_train_len, len(whole_test_data) - task_train_len))

    if args.full_meta:
        meta_len = int(args.meta_data_pcnt * len(task_train_data))
        inner_meta_data, _ = random_split(
            task_train_data, lengths=(meta_len, len(task_train_data) - meta_len))
        inner_meta_train_len = int(0.8 * len(inner_meta_data))
        inner_meta_train, inner_meta_test = random_split(
            inner_meta_data, lengths=(inner_meta_train_len,
                                        len(inner_meta_data) - inner_meta_train_len))
    else:
        inner_meta_train = None
        inner_meta_test = None

    return MetaDataset(meta_train=meta_train_data, task=task_data, task_train=task_train_data,
                       inner_meta=(inner_meta_train, inner_meta_test),
                       input_dim=None, output_dim=args.y_dim)


def get_mnist_data_tuple(args, data, train=True):
    dataset = "train" if train else "test"

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Making data tuple")

    data_path = get_path_from_args(args) / dataset

    if (os.path.exists(data_path / "x_values.npy") and os.path.exists(data_path / "s_values")
            and os.path.exists(data_path / "y_values")):
        print("data tuples found on file")
        x_all = np.load(data_path / "x_values.npy")
        s_all = pd.read_csv(data_path / "s_values", index_col=0)
        y_all = pd.read_csv(data_path / "y_values", index_col=0)
    else:
        print("data tuples haven't been created - this may take a while")
        data_loader = DataLoader(data, batch_size=args.batch_size)
        x_all, s_all, y_all = [], [], []

        for x, s, y in tqdm(data_loader):
            x_all.extend(x.numpy())
            s_all.extend(s.numpy())
            y_all.extend(y.numpy())

        x_all = np.array(x_all)
        np.save(data_path / 'x_values', x_all)
        # s_all = pd.DataFrame(np.array(s_all), columns=['sens_r', 'sens_g', 'sens_b'])
        s_all = pd.DataFrame(np.array(s_all), columns=['sens'])
        s_all.to_csv(data_path / "s_values")

        y_all = pd.DataFrame(np.array(y_all), columns=['label'])
        y_all.to_csv(data_path / "y_values")

    return DataTuple(x_all, s_all, y_all)


def load_cmnist_from_file(args):
    train_data = CMNIST(args, train=True)
    test_data = CMNIST(args, train=False, normalize_transform=train_data.normalize_transform)

    return train_data, test_data


# def save_date(args, root='../data'):
#     from torchvision.transforms import ToPILImage
#     path = Path(root) / args.dataset
#     dataloader = []
#     to_pil = ToPILImage()
#     for x, s, y in dataloader:
#         for sample in x.unfold(dim=0):
#             im = to_pil(x.detach().cpu())
#             im.save(path / , 'PNG')
