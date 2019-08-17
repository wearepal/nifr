from typing import NamedTuple
from typing import Optional

import torch
from torch.utils.data import Dataset, random_split, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

from finn.data.ld_augmentation import LdAugmentedDataset, LdColorizer
from .adult import load_adult_data
from .cmnist import ColouredMNIST


class DatasetTuple(NamedTuple):
    pretrain: Optional[Dataset]
    task: Dataset
    task_train: Dataset
    input_dim: Optional[int]
    output_dim: Optional[int]


def load_dataset(args) -> DatasetTuple:
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

        train_data = MNIST(root=args.root, download=True, train=True)
        pretrain_data, train_data = random_split(train_data, lengths=(50000, 10000))
        test_data = MNIST(root=args.root, download=True, train=False)

        colorizer = LdColorizer(scale=args.scale, background=args.background,
                                black=args.black, binarize=args.binarize)

        pretrain_data = LdAugmentedDataset(pretrain_data, ld_augmentations=colorizer,
                                           li_augmentation=True, shuffle=True)
        train_data = LdAugmentedDataset(train_data, ld_augmentations=colorizer,
                                        li_augmentation=False, shuffle=True)
        test_data = LdAugmentedDataset(test_data, ld_augmentations=colorizer,
                                       li_augmentation=True, shuffle=False)

        args.y_dim = 10
        args.s_dim = 10

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
    pretrain_len = int(args.data_pcnt * len(pretrain_data))
    pretrain_data, _ = random_split(
        pretrain_data, lengths=(pretrain_len, len(pretrain_data) - pretrain_len)
    )

    # shrink task set according to args.data_pcnt
    task_len = int(args.data_pcnt * len(test_data))
    test_data, _ = random_split(test_data, lengths=(task_len, len(test_data) - task_len))
    test_data.transform = transforms.ToTensor()
    # shrink task train set according to args.data_pcnt
    task_train_len = int(args.data_pcnt * len(whole_test_data))
    train_data, _ = random_split(
        whole_test_data, lengths=(task_train_len, len(whole_test_data) - task_train_len)
    )

    return DatasetTuple(
        pretrain=pretrain_data,
        task=test_data,
        task_train=train_data,
        input_dim=None,
        output_dim=args.y_dim,
    )
