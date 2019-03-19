import copy

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

import pandas as pd

from ethicml.algorithms.pytorch_common import CustomDataset
from ethicml.evaluators.per_sensitive_attribute import metric_per_sensitive_attribute, \
    diff_per_sensitive_attribute, ratio_per_sensitive_attribute
from ethicml.metrics import Accuracy, ProbPos
from ethicml.utility.heaviside import Heaviside



class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer = nn.Linear(input_size, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.layer(x))


def main(train=None, test=None, S_SECTION=None):
    train_data = CustomDataset(train)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=False)

    test_data = CustomDataset(test)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    model = Model(train.x.shape[1])
    optimizer_y = torch.optim.Adam(model.parameters())
    for i in tqdm(range(75)):
        for b, (embedding, sens_label, class_label) in enumerate(train_loader):
            y_pred = model(embedding)

            y_loss = F.binary_cross_entropy(y_pred, class_label)

            optimizer_y.zero_grad()
            y_loss.backward()
            optimizer_y.step()
        optimizer_y.zero_grad()

    model.eval()
    preds = []
    for embedding, sens_label, class_label in test_loader:
        preds += model(embedding).data.numpy().tolist()

    heavi = Heaviside()

    untouched_weights = copy.deepcopy(model.layer)
    for mul in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        model.layer.weight[:, -S_SECTION:] = untouched_weights.weight[:, -S_SECTION:] * mul

        preds = []
        for embedding, sens_label, class_label in test_loader:
            preds += model(embedding).data.numpy().tolist()

        preds = pd.DataFrame(preds, columns=['preds'])
        preds = pd.DataFrame(heavi.apply(preds.values), columns=['preds'])
        per_sens_dict = metric_per_sensitive_attribute(preds, test, ProbPos())
        diff = diff_per_sensitive_attribute(per_sens_dict)
        ratio = ratio_per_sensitive_attribute(per_sens_dict)
        print(f"{mul:3.1f} z(n-{S_SECTION}) -> y {per_sens_dict} \t "
              f"ratio: {list(ratio.values())[0]:.5f} \t "
              f"diff: {list(diff.values())[0]:.5f} \t "
              f"Acc: {Accuracy().score(preds['preds'], test):.5f}")


if __name__ == '__main':
    main()
