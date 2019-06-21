import copy

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import pandas as pd

from ethicml.implementations.pytorch_common import CustomDataset
from ethicml.evaluators.per_sensitive_attribute import (
    metric_per_sensitive_attribute,
    diff_per_sensitive_attribute,
    ratio_per_sensitive_attribute,
)
from ethicml.metrics import Accuracy, ProbPos
from ethicml.utility.heaviside import Heaviside

EPOCHS = 75


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer = nn.Linear(input_size, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.layer(x))


def main(train, test, zs_dim, gpu=0):
    train_data = CustomDataset(train)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=False)

    test_data = CustomDataset(test)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model = Model(train.x.shape[1]).to(device)
    optimizer_y = torch.optim.Adam(model.parameters())
    for i in tqdm(range(EPOCHS)):
        for b, (embedding, _, class_label) in enumerate(train_loader):
            embedding = embedding.to(device)
            class_label = class_label.to(device)
            optimizer_y.zero_grad()

            y_pred = model(embedding)

            y_loss = F.binary_cross_entropy(y_pred, class_label)

            y_loss.backward()
            optimizer_y.step()

    model.eval()
    preds = []
    for embedding, _, _ in test_loader:
        embedding = embedding.to(device)
        preds += model(embedding).data.cpu().numpy().tolist()

    heavi = Heaviside()

    untouched_weights = copy.deepcopy(model.layer)
    for mul in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        model.layer.weight[:, -zs_dim:] = untouched_weights.weight[:, -zs_dim:] * mul

        preds = []
        for embedding, _, _ in test_loader:
            embedding = embedding.to(device)
            preds += model(embedding).data.cpu().numpy().tolist()

        preds = pd.DataFrame(preds, columns=['preds'])
        preds = pd.DataFrame(heavi.apply(preds.values), columns=['preds'])
        per_sens_dict = metric_per_sensitive_attribute(preds, test, ProbPos())
        diff = diff_per_sensitive_attribute(per_sens_dict)
        ratio = ratio_per_sensitive_attribute(per_sens_dict)
        print(
            f"{mul:3.1f} z(n-{zs_dim}) -> y {per_sens_dict} \t "
            f"ratio: {list(ratio.values())[0]:.5f} \t "
            f"diff: {list(diff.values())[0]:.5f} \t "
            f"Acc: {Accuracy().score(preds['preds'], test):.5f}"
        )
