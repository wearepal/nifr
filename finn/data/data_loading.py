from typing import NamedTuple, Union
from typing import Optional

from ethicml.data import Adult
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from finn.data.dataset_wrappers import DataTupleDataset, LdAugmentedDataset
from finn.data.transforms import LdColorizer
from .adult import load_adult_data


class DatasetTriplet(NamedTuple):
    pretrain: Optional[Dataset]
    task: Union[Dataset, dict]
    task_train: Union[Dataset, dict]
    input_dim: Optional[int]
    output_dim: Optional[int]


def load_dataset(args) -> DatasetTriplet:
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
                                           num_classes=10,
                                           li_augmentation=True,
                                           base_augmentations=data_aug + [to_tensor])
        train_data = LdAugmentedDataset(train_data, ld_augmentations=colorizer,
                                        num_classes=10,
                                        li_augmentation=False,
                                        base_augmentations=data_aug + [to_tensor])
        test_data = LdAugmentedDataset(test_data, ld_augmentations=colorizer,
                                       num_classes=10,
                                       li_augmentation=True,
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
            pretrain_data.shrink(args.data_pcnt)
            train_data.shrink(args.data_pcnt)
            test_data.shrink(args.data_pcnt)

    return DatasetTriplet(
        pretrain=pretrain_data,
        task=test_data,
        task_train=train_data,
        input_dim=None,
        output_dim=args.y_dim,
    )
