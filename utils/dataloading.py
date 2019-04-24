import pandas as pd
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader

from ethicml.data.load import load_data
from ethicml.algorithms.utils import DataTuple
from sklearn.preprocessing import StandardScaler

from data.cmnist import CMNIST


def load_adult_data(args):
    """Load dataset from the files specified in ARGS and return it as PyTorch datasets"""

    from ethicml.data import Adult
    from ethicml.preprocessing.train_test_split import train_test_split

    data = load_data(Adult())
    train_tuple, test_tuple = train_test_split(data)

    # def load_dataframe(path: Path) -> pd.DataFrame:
    #     """Load dataframe from a parquet file"""
    #     with path.open('rb') as f:
    #         df = pd.read_feather(f)
    #     return torch.tensor(df.values, dtype=torch.float32)
    #
    # train_x = load_dataframe(Path(args.train_x))
    # train_s = load_dataframe(Path(args.train_s))
    # train_y = load_dataframe(Path(args.train_y))
    # test_x = load_dataframe(Path(args.test_x))
    # test_s = load_dataframe(Path(args.test_s))
    # test_y = load_dataframe(Path(args.test_y))

    # train_test_split()
    # train_data = TensorDataset(train_x, train_s, train_y)
    # test_data = TensorDataset(test_x, test_s, test_y)

    scaler = StandardScaler()
    train_tuple.x = scaler.fit_transform(train_tuple.x)
    test_tuple.x = scaler.transform(test_tuple.x)

    train_data = TensorDataset(*[torch.tensor(df.values, dtype=torch.float32) for df in train_tuple])
    test_data = TensorDataset(*[torch.tensor(df.values, dtype=torch.float32) for df in test_tuple])

    return train_data, test_data, train_tuple, test_tuple,


def get_mnist_data_tuple(args, data):

    data_loader = DataLoader(data, batch_size=args.batch_size)
    x_all, s_all, y_all = [], [], []

    for x, s, y in data_loader:
        x_all.extend(x.numpy())
        s_all.extend(s.numpy())
        y_all.extend(y.numpy())

    x_all = np.array(x_all)
    # s_all = pd.DataFrame(np.array(s_all), columns=['sens_r', 'sens_g', 'sens_b'])
    s_all = pd.DataFrame(np.array(s_all), columns=['sens'])

    y_all = pd.DataFrame(np.array(y_all), columns=['label'])

    return DataTuple(x_all, s_all, y_all)


def load_cmnist_from_file(args):
    train_data = CMNIST(train=True)
    test_data = CMNIST(train=False)

    train_tuple = get_mnist_data_tuple(args, train_data)
    test_tuple = get_mnist_data_tuple(args, test_data)

    return train_data, test_data, train_tuple, test_tuple


def load_dataset(args):
    if args.dataset == 'cmnist':
        train_data, test_data, train_tuple, test_tuple = load_cmnist_from_file(args)
    else:
        train_data, test_data, train_tuple, test_tuple = load_adult_data(args)

    return train_data, test_data, train_tuple, test_tuple


# def save_date(args, root='../data'):
#     from torchvision.transforms import ToPILImage
#     path = Path(root) / args.dataset
#     dataloader = []
#     to_pil = ToPILImage()
#     for x, s, y in dataloader:
#         for sample in x.unfold(dim=0):
#             im = to_pil(x.detach().cpu())
#             im.save(path / , 'PNG')
