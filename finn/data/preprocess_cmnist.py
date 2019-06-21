import os
from pathlib import Path

import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from finn.data.colorized_mnist import ColorizedMNIST


def load_cmnist_data(args):
    train_data = ColorizedMNIST(
        './data',
        download=True,
        train=True,
        scale=args.scale,
        transform=transforms.ToTensor(),
        cspace=args.cspace,
        background=args.background,
        black=args.black,
    )
    test_data = ColorizedMNIST(
        './data',
        download=True,
        train=False,
        scale=args.scale,
        transform=transforms.ToTensor(),
        cspace=args.cspace,
        background=args.background,
        black=args.black,
    )

    return train_data, test_data


def dataset_args_to_str(args):
    if args.black and args.background:
        digit_col = 'black'
        bg_col = 'col'
    elif args.black and not args.background:
        digit_col = 'col'
        bg_col = 'black'
    elif not args.black and args.background:
        digit_col = 'white'
        bg_col = 'col'
    elif not args.black and not args.background:
        digit_col = 'col'
        bg_col = 'white'

    else:
        raise NotImplementedError(
            "How is it possible to select a digit and bg color combo that got you here?"
        )

    return digit_col, bg_col


def get_path_from_args(args):
    digit_col, bg_col = dataset_args_to_str(args)
    return Path(".") / args.root / f"cmnist-sc_{args.scale}-dig_{digit_col}-bg_{bg_col}"


def make_cmnist_dataset(args):
    # Get the mnist dataset
    # Colorize the dataset

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = load_cmnist_data(args)

    # Save the dataset with class labels and sensnitive labels
    digit_col, bg_col = dataset_args_to_str(args)
    path = get_path_from_args(args)
    print(f"checking if path {path} exists")

    if not os.path.exists(path / "train"):
        print(
            f"creating training set with params:\n"
            f"\t- scale: {args.scale}\n"
            f"\t- digit colour: {digit_col}\n"
            f"\t- background col: {bg_col}"
        )
        os.makedirs(path / "train")
        for i, set in enumerate(tqdm(train_data)):
            with open(os.path.join(path / "train", str(i)), 'wb') as f:
                torch.save(set, f)
    else:
        print("trainig set already exists")

    if not os.path.exists(path / "test"):
        print(
            f"creating test set with params:\n"
            f"\t- scale: {args.scale}\n"
            f"\t- digit colour: {digit_col}\n"
            f"\t- background col: {bg_col}"
        )
        os.makedirs(path / "test")
        for i, set in enumerate(tqdm(test_data)):
            with open(os.path.join(path / "test", str(i)), 'wb') as f:
                torch.save(set, f)
    else:
        print("test set already exists")

    print("finished preparing dataset")
