from functools import partial

import torch
import os
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    verify_str_arg,
    check_integrity,
    download_file_from_google_drive,
)
from torchvision.transforms import ToTensor


class CelebA(VisionDataset):
    """Large-scale CelebFaces Attributes (CelebA) Dataset

    <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
    Adapted from torchvision.datasets to enable the loading of data triplets
    while removing superfluous (for our purposes) elements of the dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
            Defaults to ``attr``. If empty, ``None`` will be returned as target.
        sens_attr (string): Attribute to use as the sensitive attribute.
        target_attr (string): Attribute to use as the target attribute.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba"

    file_list = [
        (
            "0B7EVK8r0v71pZjFTYXZWM3FlRnM",  # File ID
            "00d2c5bc6d35e252742224ab0c1e8fcb",  # MD5 Hash
            "img_align_celeba.zip",  # Filename
        ),
        (
            "0B7EVK8r0v71pblRyaVFSWGxPY0U",
            "75e246fa4810816ffd6ee81facbd244c",
            "list_attr_celeba.txt",
        ),
        (
            "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
            "d32c9cbf5e040fd4025c592c306e6668",
            "list_eval_partition.txt",
        ),
    ]

    def __init__(
        self,
        root,
        split="train",
        sens_attr="Male",
        target_attr="Attractive",
        transform=None,
        target_transform=None,
        download=False,
    ):
        import pandas

        super(CelebA, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
        split = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(
            fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0
        )
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        attr_names = list(attr.columns)

        sens_attr = sens_attr.capitalize()
        target_attr = target_attr.capitalize()

        if sens_attr not in attr_names:
            raise ValueError(f"{sens_attr} does not exist as an attribute.")
        if target_attr not in attr_names:
            raise ValueError(f"{target_attr} does not exist as an attribute.")

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = splits[mask].index.values
        sens_attr = attr[sens_attr]
        target_attr = attr[target_attr]

        self.sens_attr = torch.as_tensor(sens_attr[mask].values)
        self.sens_attr = (self.sens_attr + 1) // 2  # map from {-1, 1} to {0, 1}

        self.target_attr = torch.as_tensor(target_attr[mask].values)
        self.target_attr = (self.target_attr + 1) // 2  # map from {-1, 1} to {0, 1}

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "img_align_celeba"))

    def download(self):
        import zipfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(
                file_id, os.path.join(self.root, self.base_folder), filename, md5
            )

        with zipfile.ZipFile(
            os.path.join(self.root, self.base_folder, "img_align_celeba.zip"), "r"
        ) as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        X = Image.open(
            os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        )
        S = self.sens_attr[index]
        target = self.target_attr[index]

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, S, target

    def __len__(self):
        return len(self.sens_attr)

    def extra_repr(self):
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_data = CelebA(
        root=r"..\..\..\datasets", split="train", download=False, transform=ToTensor()
    )

    train_loader = DataLoader(train_data, batch_size=9)
    x, s, y = next(iter(train_loader))
    assert x.size(0) == 9
    assert x.size(0) == s.size(0) == y.size(0)
