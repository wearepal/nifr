from typing import NamedTuple, Optional

from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, KMNIST
from ethicml.data import create_genfaces_dataset
from ethicml.vision.data import LdColorizer

from nosinn.configs import SharedArgs

from .adult import load_adult_data
from .celeba import CelebA
from .dataset_wrappers import LdAugmentedDataset
from .misc import shrink_dataset, train_test_split
from .perturbed_adult import load_perturbed_adult
from .ssrp import SSRP
from .transforms import NoisyDequantize, Quantize

__all__ = ["DatasetTriplet", "load_dataset"]


class DatasetTriplet(NamedTuple):
    pretrain: Optional[Dataset]
    task: Dataset
    task_train: Dataset
    s_dim: int
    y_dim: int


def load_dataset(args: SharedArgs) -> DatasetTriplet:
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
        if args.quant_level != "8":
            base_aug.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            base_aug.append(NoisyDequantize(int(args.quant_level)))
        train_data = MNIST(root=args.root, download=True, train=True)

        pretrain_len = round(args.pretrain_pcnt * len(train_data))
        train_len = len(train_data) - pretrain_len
        pretrain_data, train_data = random_split(train_data, lengths=(pretrain_len, train_len))

        test_data = MNIST(root=args.root, download=True, train=False)

        pretrain_data = train_test_split(
            KMNIST(root=args.root, download=True, train=True), args.pretrain_pcnt
        )
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

        args.y_dim = 10
        args.s_dim = 10

    elif args.dataset == "ssrp":
        image_size = 64
        transform = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        if args.quant_level != "8":
            transform.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            transform.append(NoisyDequantize(int(args.quant_level)))
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        pretrain_data = SSRP(args.root, pretrain=True, download=True, transform=transform)
        train_test_data = SSRP(args.root, pretrain=False, download=True, transform=transform)

        train_data, test_data = train_test_split(train_test_data, train_pcnt=(1 - args.test_pcnt))

        args.y_dim = train_test_data.num_classes
        args.s_dim = pretrain_data.num_classes

    elif args.dataset == "celeba":

        image_size = 64
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        if args.quant_level != "8":
            transform.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            transform.append(NoisyDequantize(int(args.quant_level)))
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        unbiased_pcnt = args.test_pcnt + args.pretrain_pcnt
        unbiased_data = CelebA(
            root=args.root,
            sens_attrs=args.celeba_sens_attr,
            target_attr_name=args.celeba_target_attr,
            biased=False,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        pretrain_len = round(args.pretrain_pcnt / unbiased_pcnt * len(unbiased_data))
        test_len = len(unbiased_data) - pretrain_len
        pretrain_data, test_data = random_split(unbiased_data, lengths=(pretrain_len, test_len))

        train_data = CelebA(
            root=args.root,
            sens_attrs=args.celeba_sens_attr,
            target_attr_name=args.celeba_target_attr,
            biased=True,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        args.y_dim = 1
        args.s_dim = unbiased_data.s_dim

    elif args.dataset == "genfaces":

        image_size = 64
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        if args.quant_level != "8":
            transform.append(Quantize(int(args.quant_level)))
        if args.input_noise:
            transform.append(NoisyDequantize(int(args.quant_level)))
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        unbiased_pcnt = args.test_pcnt + args.pretrain_pcnt
        unbiased_data = create_genfaces_dataset(
            root=args.root,
            sens_attr_name=args.genfaces_sens_attr,
            target_attr_name=args.genfaces_target_attr,
            biased=False,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        pretrain_len = round(args.pretrain_pcnt / unbiased_pcnt * len(unbiased_data))
        test_len = len(unbiased_data) - pretrain_len
        pretrain_data, test_data = random_split(unbiased_data, lengths=(pretrain_len, test_len))

        train_data = create_genfaces_dataset(
            root=args.root,
            sens_attr_name=args.genfaces_sens_attr,
            target_attr_name=args.genfaces_target_attr,
            biased=True,
            mixing_factor=args.task_mixing_factor,
            unbiased_pcnt=unbiased_pcnt,
            download=True,
            transform=transform,
            seed=args.data_split_seed,
        )

        args.y_dim = 1
        args.s_dim = unbiased_data.s_dim

    elif args.dataset == "adult":
        if args.input_noise:
            pretrain_data, test_data, train_data = load_perturbed_adult(args)
        else:
            pretrain_data, test_data, train_data = load_adult_data(args)

        args.y_dim = 1
        args.s_dim = 1
    else:
        raise ValueError("Invalid choice of dataset.")

    if 0 < args.data_pcnt < 1:
        pretrain_data = shrink_dataset(pretrain_data, args.data_pcnt)
        train_data = shrink_dataset(train_data, args.data_pcnt)
        test_data = shrink_dataset(test_data, args.data_pcnt)

    return DatasetTriplet(
        pretrain=pretrain_data,
        task=test_data,
        task_train=train_data,
        s_dim=args.s_dim,
        y_dim=args.y_dim,
    )
