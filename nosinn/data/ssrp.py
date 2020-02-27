from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

__all__ = ["SSRP"]


class SSRP(Dataset):
    _FILE_ID = "1RE4srtC63VnyU0e1qx16QNdjyyQXg2hj"
    _FILENAME = "ghaziabad.zip"

    def __init__(
        self,
        root: str,
        pretrain: bool,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self._base_folder = Path(root) / "ssrp"

        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted." + " You can use download=True to download it"
            )

        path_to_data = self._base_folder / "Ghaziabad"
        self.pretrain = pretrain
        if self.pretrain:
            path_to_data = path_to_data / "Pre_Train"
        else:
            path_to_data = path_to_data / "Task"

        self._dataset = ImageFolder(
            path_to_data, transform=transform, target_transform=target_transform
        )
        self.num_classes = len(self._dataset.classes)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        from torchvision.datasets.utils import download_file_from_google_drive
        import zipfile

        download_file_from_google_drive(self._FILE_ID, self._base_folder, self._FILENAME)
        with zipfile.ZipFile(self._base_folder / self._FILENAME, "r") as f:
            f.extractall(self._base_folder)

    def _check_integrity(self) -> None:
        return (self._base_folder / "Ghaziabad").is_dir()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[Tensor, Tensor, None], Tuple[Tensor, None, Tensor]]:
        if self.pretrain:
            x, s = self._dataset[index]
            y = torch.tensor([])
        else:
            x, y = self._dataset[index]
            s = torch.tensor([])
        return x, s, y
