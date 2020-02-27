import os
import shutil

import pandas as pd
from torchvision import transforms


def format_data_folder(root: str):
    data = f"{root}/data"
    train_dir = f"{data}/train"

    bright = pd.read_csv(f"{root}/list_bright.txt", header=None)
    dark = pd.read_csv(f"{root}/list_dark.txt", header=None)

    bright_dogs = bright[bright[0].str.contains("dog")]
    bright_cats = bright[bright[0].str.contains("cat")]
    dark_dogs = dark[dark[0].str.contains("dog")]
    dark_cats = dark[dark[0].str.contains("cat")]

    bright_dogs_dark_cats = pd.concat([bright_dogs, dark_cats], axis=0, sort=False)
    dark_dogs_bright_cats = pd.concat([dark_dogs, bright_cats], axis=0, sort=False)

    bright_dogs_dark_cats_dir = f"{data}/bright_dogs_dark_cats"
    dark_dogs_bright_cats_dir = f"{data}/dark_dogs_bright_cats"

    if not os.path.exists(bright_dogs_dark_cats_dir):
        os.makedirs(bright_dogs_dark_cats_dir)
    if not os.path.exists(dark_dogs_bright_cats_dir):
        os.makedirs(dark_dogs_bright_cats_dir)

    if not os.path.exists(f"{bright_dogs_dark_cats_dir}/dog"):
        os.makedirs(f"{bright_dogs_dark_cats_dir}/dog")
    if not os.path.exists(f"{bright_dogs_dark_cats_dir}/cat"):
        os.makedirs(f"{bright_dogs_dark_cats_dir}/cat")

    dog = f"{bright_dogs_dark_cats_dir}/dog"
    cat = f"{bright_dogs_dark_cats_dir}/cat"
    for file in bright_dogs_dark_cats[0]:
        if "dog" in file:
            dst = dog
        else:
            dst = cat
        shutil.copy(f"{train_dir}/{file}", dst)

    if not os.path.exists(f"{dark_dogs_bright_cats_dir}/dog"):
        os.makedirs(f"{dark_dogs_bright_cats_dir}/dog")
    if not os.path.exists(f"{dark_dogs_bright_cats_dir}/cat"):
        os.makedirs(f"{dark_dogs_bright_cats_dir}/cat")

    dog = f"{dark_dogs_bright_cats_dir}/dog"
    cat = f"{dark_dogs_bright_cats_dir}/cat"
    for file in dark_dogs_bright_cats[0]:
        if "dog" in file:
            dst = dog
        else:
            dst = cat
        shutil.copy(f"{train_dir}/{file}", dst)


def load_dvc_data(root: str):
    from torchvision.datasets import ImageFolder

    tforms = transforms.Compose(
        [
            transforms.Resize(size=[224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train = ImageFolder(root=f"{root}/bright_dogs_dark_cats", transform=tforms)
    test = ImageFolder(root=f"{root}/dark_dogs_bright_cats", transform=tforms)

    return train, test
