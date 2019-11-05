from typing import NamedTuple, Optional

from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from ethicml.data import Adult

from .dataset_wrappers import DataTupleDataset, LdAugmentedDataset
from .transforms import LdColorizer, NoisyDequantize, Quantize
from .adult import load_adult_data
from .perturbed_adult import load_perturbed_adult
from .celeba import CelebA


class DatasetTriplet(NamedTuple):
    pretrain: Optional[Dataset]
    task: Dataset
    task_train: Dataset
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None


def load_dataset(args) -> DatasetTriplet:
    assert args.pretrain
    pretrain_data: Dataset
    test_data: Dataset
    train_data: Dataset

    # =============== get whole dataset ===================
    if args.dataset == "cmnist":
        base_aug = [transforms.ToTensor()]
        data_aug = []
        if args.rotate_data:
            data_aug.append(transforms.RandomAffine(degrees=15))
        if args.shift_data:
            data_aug.append(transforms.RandomAffine(degrees=0, translate=(0.11, 0.11)))
        if args.padding > 0:
            base_aug.insert(0, transforms.Pad(args.padding))
        if args.quant_level != 8:
            base_aug.append(Quantize(args.quant_level))
        if args.input_noise:
            base_aug.append(NoisyDequantize(args.quant_level))
        train_data = MNIST(root=args.root, download=True, train=True)

        pretrain_len = round(args.pretrain_pcnt * len(train_data))
        train_len = len(train_data) - pretrain_len
        pretrain_data, train_data = random_split(train_data, lengths=(pretrain_len, train_len))

        test_data = MNIST(root=args.root, download=True, train=False)

        colorizer = LdColorizer(
            scale=args.scale,
            background=args.background,
            black=args.black,
            binarize=args.binarize,
            greyscale=args.greyscale,
        )

        pretrain_data = LdAugmentedDataset(
            pretrain_data,
            ld_augmentations=colorizer,
            num_classes=10,
            li_augmentation=True,
            base_augmentations=data_aug + base_aug,
        )
        train_data = LdAugmentedDataset(
            train_data,
            ld_augmentations=colorizer,
            num_classes=10,
            li_augmentation=False,
            base_augmentations=data_aug + base_aug,
        )
        test_data = LdAugmentedDataset(
            test_data,
            ld_augmentations=colorizer,
            num_classes=10,
            li_augmentation=True,
            base_augmentations=base_aug,
        )

        if 0 < args.data_pcnt < 1:
            pretrain_data.subsample(args.data_pcnt)
            train_data.subsample(args.data_pcnt)
            test_data.subsample(args.data_pcnt)

        args.y_dim = 10
        args.s_dim = 10

    elif args.dataset == "celeba":

        image_size = 64
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        if args.quant_level != 8:
            transform.append(Quantize(args.quant_level))
        if args.input_noise:
            transform.append(NoisyDequantize(args.quant_level))

        transform = transforms.Compose(transform)

        unbiased_pcnt = args.task_pcnt + args.pretrain_pcnt
        unbiased_data = CelebA(
            root=args.root,
            biased=False,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        pretrain_len = round(args.pretrain_pcnt * len(unbiased_data))
        test_len = len(unbiased_data) - pretrain_len
        pretrain_data, test_data = random_split(unbiased_data, lengths=(pretrain_len, test_len))

        train_data = CelebA(
            root=args.root,
            biased=True,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        args.y_dim = 1
        args.s_dim = 1

    elif args.dataset == "adult":
        if args.input_noise:
            pretrain_data, test_data, train_data = load_perturbed_adult(args)
        else:
            pretrain_data, test_data, train_data = load_adult_data(args)

        args.y_dim = 1
        args.s_dim = 1

        if 0 < args.data_pcnt < 1:
            pretrain_data.shrink(args.data_pcnt)
            train_data.shrink(args.data_pcnt)
            test_data.shrink(args.data_pcnt)
    else:
        raise ValueError("Invalid choice of dataset.")

    return DatasetTriplet(
        pretrain=pretrain_data,
        task=test_data,
        task_train=train_data,
        input_dim=None,
        output_dim=args.y_dim,
    )
