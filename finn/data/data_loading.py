from typing import NamedTuple
from typing import Optional

from ethicml.data import Adult
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from finn.data.datasets import DataTupleDataset, LdAugmentedDataset
from finn.data.ld_augmentation import LdColorizer
from finn.data.misc import shrink_dataset
from .adult import load_adult_data


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
        to_tensor = transforms.ToTensor()
        data_aug = []
        if args.rotate_data:
            data_aug.append(transforms.RandomAffine(degrees=15))
        if args.shift_data:
            data_aug.append(transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)))

        train_data = MNIST(root=args.root, download=True, train=True)
        pretrain_data, train_data = random_split(train_data, lengths=(50000, 10000))
        test_data = MNIST(root=args.root, download=True, train=False)

        colorizer = LdColorizer(scale=args.scale, background=args.background,
                                black=args.black, binarize=args.binarize)

        pretrain_data = LdAugmentedDataset(pretrain_data, ld_augmentations=colorizer,
                                           li_augmentation=True, shuffle=True,
                                           base_augmentations=data_aug + [to_tensor])
        train_data = LdAugmentedDataset(train_data, ld_augmentations=colorizer,
                                        li_augmentation=False, shuffle=True,
                                        base_augmentations=data_aug + [to_tensor])
        test_data = LdAugmentedDataset(test_data, ld_augmentations=colorizer,
                                       li_augmentation=True, shuffle=False,
                                       base_augmentations=[to_tensor])

        if 0 < args.data_pcnt < 1:
            pretrain_data.subsample(args.data_pcnt)
            train_data.subsample(args.data_pcnt)
            test_data.subsample(args.data_pcnt)

        args.y_dim = 10
        args.s_dim = 10

    else:
        pretrain_tuple, test_tuple, train_tuple = load_adult_data(args)
        source_dataset = Adult()
        pretrain_data = DataTupleDataset(pretrain_tuple, source_dataset)
        train_data = DataTupleDataset(train_tuple, source_dataset)
        test_data = DataTupleDataset(test_tuple, source_dataset)

        args.y_dim = 1
        args.s_dim = 1

        if 0 < args.data_pcnt < 1:
            pretrain_data = shrink_dataset(pretrain_data, args.data_pcnt)
            train_data = shrink_dataset(train_data, args.data_pcnt)
            test_data = shrink_dataset(test_data, args.data_pcnt)

    return DatasetTuple(
        pretrain=pretrain_data,
        task=test_data,
        task_train=train_data,
        input_dim=None,
        output_dim=args.y_dim,
    )
