import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _convert():
    """Convert the two npy files to a single (compressed) npz file"""
    gt = np.load("imdb_age_gender.npy", encoding="latin1").item()
    sp = np.load("imdb_split.npy", encoding="latin1").item()

    train_files = sp["train_list"]
    train_age = []
    train_gender = []
    for i in range(len(train_files)):
        labels = gt[train_files[i].encode("utf-8")]
        train_age.append(labels["age"])
        train_gender.append(labels["gender"])

    test_files = sp["test_list"]
    test_age = []
    test_gender = []
    for i in range(len(test_files)):
        labels = gt[test_files[i].encode("utf-8")]
        test_age.append(labels["age"])
        test_gender.append(labels["gender"])

    data = dict(
        train_gender=np.array(train_gender),
        train_age=np.array(train_age),
        train_files=np.array(train_files),
        test_gender=np.array(test_gender),
        test_age=np.array(test_age),
        test_files=np.array(test_files),
    )

    np.savez_compressed("imdb", **data)


class ImdbData(Dataset):

    _sens_attributes = ["age", "gender"]

    def __init(
        self, path_to_biased_data, image_data_root, train=True, sens_attr="gender", transform=None
    ):

        self.transform = transform or None

        if sens_attr not in self._sens_attributes:
            raise ValueError(f"{sens_attr} not a valid sensitive attribute.")

        self.image_data_root = image_data_root

        with open(path_to_biased_data) as f:
            data = np.load(f)
            prefix = "train" if train else "test"

            gender = data[f"{prefix}_gender"]
            age = data[f"{prefix}_age"]

            if sens_attr == "gender":
                self.sens_data = gender
                self.target_data = age
            else:
                self.sens_attr = age
                self.target_data = gender

            self.files = data[f"{prefix}_files"]

    def __getitem__(self, index):

        fname = self.files[index]
        path = os.path.join(self.image_data_root, fname)
        img = Image.open(path)

        if self.transform is not None:
            self.transform(img)

        sens = self.sens_data[index]
        target = self.target_data[index]

        return img, sens, target
