from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from finn.models.base import BaseModel


class Classifier(BaseModel):
    """ Wrapper for classifier models.
    """

    def __init__(
        self,
        model,
        n_classes: int,
        optimizer_args: dict = None,
    ) -> None:
        """Build classifier model.

        Args:).
            n_classes: Positive integer. Number of class labels.
            model: nn.Module. Classifier model to wrap around.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """
        if n_classes < 2:
            raise ValueError(f"Invalid number of classes: must equal 2 or more,"
                             f" {n_classes} given.")
        if n_classes == 2:
            self.criterion = "bce"
        else:
            self.criterion = "ce"

        self.out_dim = n_classes if self.criterion == 'ce' else 1

        super().__init__(
            model,
            optimizer_args=optimizer_args,
        )

    def _apply_criterion(self, logits, targets):

        if self.criterion == "bce":
            if targets.dtype != torch.float32:
                targets = targets.float()
            logits = logits.view(-1, 1)
            targets = targets.view(-1, 1)
            return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        else:
            targets = targets.view(-1)
            if targets.dtype != torch.long:
                targets = targets.long()
            return F.cross_entropy(logits, targets, reduction="none")

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Make prediction.

        Args:
            inputs: Tensor. Inputs to the classifier.

        Returns:
            Class predictions (tensor) for the given data samples.
        """
        outputs = super().__call__(inputs)
        predicted = torch.argmax(outputs, dim=1)

        return predicted

    def predict_dataset(self, data, device, batch_size=100):
        if not isinstance(data, DataLoader):
            data = DataLoader(data, batch_size=batch_size,
                              shuffle=False, pin_memory=True)
        preds, actual, sens = [], [], []
        with torch.set_grad_enabled(False):
            for x, s, y in data:
                x = x.to(device)
                y = y.to(device)

                batch_preds = self.predict(x)
                preds.append(batch_preds)
                actual.append(y)
                sens.append(s)

        preds = torch.cat(preds, dim=0).cpu().detach().view(-1)
        actual = torch.cat(actual, dim=0).cpu().detach().view(-1)
        sens = torch.cat(sens, dim=0).cpu().detach().view(-1)

        return preds, actual, sens

    def compute_accuracy(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        labeled: torch.Tensor,
        top: int = 1,
    ) -> float:
        """Computes the classification accuracy.

        Args:
            outputs: Tensor. Classifier outputs.
            targets: Tensor. Targets for each input.
            labeled: Tensor. Binary variable indicating whether a target exists.
            top (int): Top-K accuracy.

        Returns:
            Accuracy of the predictions (float).
        """

        if self.criterion == 'bce':
            pred = torch.round(outputs.sigmoid())
        else:
            _, pred = outputs.topk(top, 1, True, True)
        pred = pred.t().to(targets.dtype)
        correct = labeled.float() * pred.eq(targets.view(1, -1).expand_as(pred)).float()
        correct = correct[:top].view(-1).float().sum(0, keepdim=True)
        accuracy = correct / labeled.float().sum() * 100

        return accuracy.detach().item()

    def routine(
        self, data: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Classifier routine.

        Args:
            data: Tensor. Input data to the classifier.
            targets: Tensor. Prediction targets.
            criterion: Callable. Loss function.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        outputs = super().__call__(data)
        unlabeled = targets.eq(-1).to(targets.dtype)
        losses = self._apply_criterion(outputs, (1 - unlabeled) * targets)
        labeled = 1.0 - unlabeled
        loss = (losses * labeled.float()).sum() / labeled.float().sum()

        if labeled.sum() > 0:
            acc = self.compute_accuracy(outputs, targets, labeled)
        else:
            acc = None

        return loss, acc

    def fit(self, train_data, epochs, device, test_data=None,
            pred_s=False, batch_size=100, test_batch_size=1000):

        if not isinstance(train_data, DataLoader):
            train_data = DataLoader(train_data, batch_size=batch_size,
                                    shuffle=True, pin_memory=True)

        if test_data is not None:
            if not isinstance(train_data, DataLoader):
                train_data = DataLoader(train_data, batch_size=test_batch_size,
                                        shuffle=False, pin_memory=True)

        for epoch in range(epochs):
            print(f"===> Epoch {epoch} of classifier training")

            for x, s, y in train_data:

                if pred_s:
                    target = s
                else:
                    target = y
                x = x.to(device)
                target = target.to(device)

                self.zero_grad()
                loss, acc = self.routine(x, target)
                loss.backward()
                self.step()

            save_image(x, filename='train.png')

            if test_data is not None:
                print(f"===> Testing classifier")
                avg_test_acc = 0.0
                with torch.set_grad_enabled(False):
                    for x, s, y in test_data:

                        if pred_s:
                            target = s
                        else:
                            target = y

                        x = x.to(device)
                        target = target.to(device)
                        loss, acc = self.routine(x, target)
                        avg_test_acc += acc

                save_image(x, filename='test.png')

                avg_test_acc /= len(test_data)
                print(f"Average test accuracy: {avg_test_acc:.2f}")


