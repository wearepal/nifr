from typing import NamedTuple
from typing import Optional

import torch
from torch.utils.data import Dataset, random_split, TensorDataset
from torchvision import transforms

from .adult import load_adult_data
from .cmnist import ColouredMNIST


class DatasetWrapper(NamedTuple):
    pretrain: Optional[Dataset]
    task: Dataset
    task_train: Dataset
    input_dim: Optional[int]
    output_dim: Optional[int]


def load_dataset(args) -> DatasetWrapper:
    assert args.pretrain

    # =============== get whole dataset ===================
    if args.dataset == 'cmnist':
        cmnist_transforms = []
        if args.rotate_data:
            cmnist_transforms.append(transforms.RandomAffine(degrees=15))
        if args.shift_data:
            cmnist_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)))

        cmnist_transforms.append(transforms.ToTensor())
        cmnist_transforms = transforms.Compose(cmnist_transforms)

        whole_train_data = ColouredMNIST(
            args.root,
            use_train_split=True,
            assign_color_randomly=True,
            download=True,
            transform=cmnist_transforms,
            scale=args.scale,
            background=args.background,
            black=args.black,
            binarize=args.binarize,
        )
        whole_test_data = ColouredMNIST(
            args.root,
            use_train_split=False,
            assign_color_randomly=False,
            download=True,
            transform=cmnist_transforms,
            scale=args.scale,
            background=args.background,
            black=args.black,
            binarize=args.binarize,
        )

        # train_data, test_data = load_cmnist_from_file(args)
        args.y_dim = 10
        args.s_dim = 10

        # split the training set to get training and validation sets
        whole_train_data, whole_val_data = random_split(whole_train_data, lengths=(50000, 10000))

    else:
        meta_tuple, task_tuple, task_train_tuple = load_adult_data(args)
        whole_train_data = TensorDataset(
            *[torch.tensor(df.values, dtype=torch.float32) for df in meta_tuple]
        )
        whole_val_data = TensorDataset(
            *[torch.tensor(df.values, dtype=torch.float32) for df in task_tuple]
        )
        whole_test_data = TensorDataset(
            *[torch.tensor(df.values, dtype=torch.float32) for df in task_train_tuple]
        )
        args.y_dim = 1
        args.s_dim = 1

    # shrink meta train set according to args.data_pcnt
    pretrain_len = int(args.data_pcnt * len(whole_train_data))
    pretrain_data, _ = random_split(
        whole_train_data, lengths=(pretrain_len, len(whole_train_data) - pretrain_len)
    )

    # shrink task set according to args.data_pcnt
    task_len = int(args.data_pcnt * len(whole_val_data))
    task_data, _ = random_split(whole_val_data, lengths=(task_len, len(whole_val_data) - task_len))
    task_data.transform = transforms.ToTensor()
    # shrink task train set according to args.data_pcnt
    task_train_len = int(args.data_pcnt * len(whole_test_data))
    task_train_data, _ = random_split(
        whole_test_data, lengths=(task_train_len, len(whole_test_data) - task_train_len)
    )

    return DatasetWrapper(
        pretrain=pretrain_data,
        task=task_data,
        task_train=task_train_data,
        input_dim=None,
        output_dim=args.y_dim,
    )
