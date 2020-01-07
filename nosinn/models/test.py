import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from nosinn.data import LdColorizer
from nosinn.data.dataset_wrappers import LdAugmentedDataset
from nosinn.data.misc import shrink_dataset
from nosinn.models import Classifier
import torch.nn.functional as F

train = MNIST(root="data", download=True, train=True, transform=ToTensor())
train = shrink_dataset(train, pcnt=0.1)
test = MNIST(root="data", download=True, train=False, transform=ToTensor())
test = shrink_dataset(test, pcnt=0.1)

augment = LdColorizer(black=True, background=False, scale=0)
train = LdAugmentedDataset(
    source_dataset=train, ld_augmentations=augment, li_augmentation=False, num_classes=10
)

test = LdAugmentedDataset(
    source_dataset=test, ld_augmentations=augment, li_augmentation=True, num_classes=10
)

train = DataLoader(train, batch_size=256, pin_memory=True)
test = DataLoader(test, batch_size=256, pin_memory=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


clf = Net()
# clf = mp_28x28_net(input_dim=3, target_dim=10)
clf: Classifier = Classifier(clf, num_classes=10)


clf.fit(train_data=train, test_data=test, epochs=30, device=torch.device("cpu"), pred_s=False)
