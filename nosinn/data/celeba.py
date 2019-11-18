import os
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive
from torchvision.transforms import ToTensor

from ethicml.preprocessing import get_biased_subset, SequentialSplit
from ethicml.utility import DataTuple


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
        biased,
        mixing_factor,
        unbiased_pcnt,
        sens_attr: str = "Young",
        target_attr: str = "Smiling",
        transform=None,
        target_transform=None,
        download=False,
        seed=42,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        base = Path(self.root) / self.base_folder
        partition_file = base / "list_eval_partition.txt"
        # partition: information about which samples belong to train, val or test
        partition = pd.read_csv(
            partition_file, delim_whitespace=True, header=None, index_col=0, names=["partition"]
        )
        # raw_attr: all attributes with filenames as index
        raw_attr = pd.read_csv(base / "list_attr_celeba.txt", delim_whitespace=True, header=1)
        attr = pd.concat([partition, raw_attr], axis="columns", sort=False)
        # the filenames are used for indexing; here we turn them into a regular column
        attr = attr.reset_index(drop=False).rename(columns={"index": "filenames"})

        attr_names = list(attr.columns)

        sens_attr = sens_attr.capitalize()
        target_attr = target_attr.capitalize()

        if sens_attr not in attr_names:
            raise ValueError(f"{sens_attr} does not exist as an attribute.")
        if target_attr not in attr_names:
            raise ValueError(f"{target_attr} does not exist as an attribute.")

        filename = attr[["filenames"]]
        sens_attr = attr[[sens_attr]]
        sens_attr = (sens_attr + 1) // 2  # map from {-1, 1} to {0, 1}
        target_attr = attr[[target_attr]]
        target_attr = (target_attr + 1) // 2  # map from {-1, 1} to {0, 1}

        all_dt = DataTuple(x=filename, s=sens_attr, y=target_attr)

        # NOTE: the sequential split does not shuffle
        unbiased_dt, biased_dt, _ = SequentialSplit(train_percentage=unbiased_pcnt)(all_dt)

        if biased:
            biased_dt, _ = get_biased_subset(
                data=biased_dt, mixing_factor=mixing_factor, unbiased_pcnt=0, seed=seed
            )
            filename, sens_attr, target_attr = biased_dt
        else:
            filename, sens_attr, target_attr = unbiased_dt

        self.filename = filename.to_numpy()[:, 0]
        self.sens_attr = torch.as_tensor(sens_attr.to_numpy())

        self.target_attr = torch.as_tensor(target_attr.to_numpy())

    def _check_integrity(self):
        base = Path(self.root) / self.base_folder
        for (_, md5, filename) in self.file_list:
            fpath = base / filename
            ext = fpath.suffix
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext not in [".zip", ".7z"] and not check_integrity(str(fpath), md5):
                return False

        # Should check a hash of the images
        return (base / "img_align_celeba").is_dir()

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
    from torch.utils.data import DataLoader, random_split

    train_data = CelebA(
        root=r"./data",
        biased=True,
        mixing_factor=1.0,
        unbiased_pcnt=0.4,
        download=False,
        transform=ToTensor(),
    )

    split_size = round(0.4 * len(train_data))
    train_data, _ = random_split(train_data, lengths=(split_size, len(train_data) - split_size))
    assert len(train_data) == split_size
    train_loader = DataLoader(train_data, batch_size=9)
    x, s, y = next(iter(train_loader))

    assert x.size(0) == 9
    assert x.size(0) == s.size(0) == y.size(0)
