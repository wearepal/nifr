import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from nifr.data import load_dataset
from nifr.utils import random_seed

__all__ = ["main"]


def restricted_float(x):
    x = float(x)
    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def parse_arguments(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["adult", "cmnist"], default="cmnist")
    parser.add_argument(
        "--data-pcnt",
        type=restricted_float,
        metavar="P",
        default=1.0,
        help="data %% should be a real value > 0, and up to 1",
    )
    parser.add_argument(
        "--task-mixing-factor",
        type=float,
        metavar="P",
        default=0.0,
        help="How much of meta train should be mixed into task train?",
    )
    parser.add_argument("--pretrain", type=eval, default=True, choices=[True, False])
    parser.add_argument("--pretrain-pcnt", type=float, default=0.4)
    parser.add_argument("--test-pcnt", type=float, default=0.2)

    # Colored MNIST settings
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("-bg", "--background", type=eval, default=False, choices=[True, False])
    parser.add_argument("--black", type=eval, default=True, choices=[True, False])
    parser.add_argument("--binarize", type=eval, default=True, choices=[True, False])
    parser.add_argument("--rotate-data", type=eval, default=False, choices=[True, False])
    parser.add_argument("--shift-data", type=eval, default=False, choices=[True, False])
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--quant-level", type=int, default=8)
    parser.add_argument("--input-noise", type=eval, default=True, choices=[True, False])

    parser.add_argument("--prior-dist", type=str, default="normal", choices=["logistic", "normal"])
    parser.add_argument("--root", type=str, default="data")

    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--dims", type=str, default="100-100")
    parser.add_argument("--glow", type=eval, default=True, choices=[True, False])
    parser.add_argument("--batch-norm", type=eval, default=True, choices=[True, False])
    parser.add_argument("--bn-lag", type=float, default=0)
    parser.add_argument("--disc-hidden-dims", type=int, default=256)

    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--disc-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-split-seed", type=int, default=888)

    parser.add_argument("--results-csv", default="")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save", type=str, default="experiments/finn")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--super-val",
        type=eval,
        default=False,
        choices=[True, False],
        help="Train classifier on encodings as part of validation step.",
    )
    parser.add_argument("--val-freq", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=10)

    parser.add_argument("--zs-frac", type=float, default=0.33)
    parser.add_argument("--zy-frac", type=float, default=0.33)

    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--log-px-weight", type=float, default=1.0e-3)
    parser.add_argument("-pyzyw", "--pred-y-weight", type=float, default=0.0)
    parser.add_argument("-pszyw", "--pred-s-from-zy-weight", type=float, default=1.0)
    parser.add_argument("-pszsw", "--pred-s-from-zs-weight", type=float, default=0.0)
    parser.add_argument(
        "-elw",
        "--entropy-loss-weight",
        type=float,
        default=0.0,
        help="Weight of the entropy loss for the adversarial discriminator",
    )
    parser.add_argument(
        "--use-s",
        type=eval,
        default=False,
        choices=[True, False],
        help="Use s as input (if s is a separate feature)",
    )
    parser.add_argument("--spectral-norm", type=eval, default=False, choices=[True, False])
    parser.add_argument("--proj-grads", type=eval, default=True, choices=[True, False])
    # classifier parameters (for computing fairness metrics)
    parser.add_argument("--mlp-clf", type=eval, default=False, choices=[True, False])
    parser.add_argument("--clf-epochs", type=int, metavar="N", default=50)
    parser.add_argument("--clf-early-stopping", type=int, metavar="N", default=20)
    parser.add_argument("--clf-val-ratio", type=float, metavar="R", default=0.2)
    parser.add_argument("--clf-reg-weight", type=float, metavar="R", default=1.0e-7)

    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use (if available)")
    parser.add_argument(
        "--use-comet",
        type=eval,
        default=False,
        choices=[True, False],
        help="whether to use the comet.ml logging",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Gamma value for Exponential Learning Rate scheduler. "
        "Value of 0.95 arbitrarily chosen.",
    )
    parser.add_argument(
        "--meta-learn",
        type=eval,
        default=True,
        choices=[True, False],
        help="Use meta learning procedure",
    )
    parser.add_argument("--drop-native", type=eval, default=True, choices=[True, False])

    parser.add_argument("--task-pcnt", type=float, default=0.2)
    parser.add_argument("--meta-pcnt", type=float, default=0.4)

    parser.add_argument("--drop-discrete", type=eval, default=False)
    parser.add_argument("--save-to-csv", type=eval, default=False, choices=[True, False])

    parser.add_argument("--save-dir", type=str, default="./")
    parser.add_argument("--data-dir", type=str, default="./data/ln2ldata/")
    parser.add_argument("--exp-name", type=str, default="experiments/ln2l")
    parser.add_argument("--train-baseline", type=eval, default=False, choices=[True, False])

    parser.add_argument("--save-step", type=int, default=10)
    parser.add_argument("--log-step", type=int, default=50)
    parser.add_argument("--max-step", type=int, default=100)

    parser.add_argument("--checkpoint", default=None, help="checkpoint to resume")
    parser.add_argument(
        "--use-pretrain",
        type=eval,
        default=False,
        choices=[False, True],
        help="whether it use pre-trained parameters if exists",
    )
    parser.add_argument(
        "--eval-only",
        type=eval,
        default=False,
        choices=[False, True],
        help="Skip the training step?",
    )
    parser.add_argument(
        "--use-ln2l-data",
        type=eval,
        default=True,
        choices=[False, True],
        help="use the 'Learning not to learn' data? if False, use ours",
    )
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args(raw_args)


def main(raw_args=None):
    """
    BEFORE RUNNING - MAKE SURE THE EXPERIMENT DIRECTORY EXISTS
    :param raw_args:
    :return:
    """

    args = parse_arguments(raw_args)
    args.exp_name += "_baseline" if args.train_baseline else ""
    pth = Path(args.save_dir) / args.exp_name
    pth.mkdir(exist_ok=True, parents=True)

    use_gpu = torch.cuda.is_available() and not args.gpu < 0
    random_seed(args.seed, use_gpu)

    if args.use_ln2l_data:
        args.data_split = "train"
        custom_loader = WholeDataLoader(args)
        trainval_loader = torch.utils.data.DataLoader(
            custom_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
    else:
        our_datasets = load_dataset(args)
        trainval_loader = torch.utils.data.DataLoader(
            our_datasets.task_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    trainer = Trainer(args)

    trainer.option.is_train = True
    if not args.eval_only:
        trainer.train(trainval_loader)

    trainer.option.is_train = False
    trainer.option.checkpoint = pth / "checkpoint_step_0099.pth"

    print("Performance on training set")
    trainer._validate(trainval_loader)

    if args.use_ln2l_data:
        args.data_split = "test"
        custom_loader = WholeDataLoader(args)
        testval_loader = torch.utils.data.DataLoader(
            custom_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
    else:
        testval_loader = torch.utils.data.DataLoader(
            our_datasets.task,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    print("Performance on the test set")
    trainer._validate(testval_loader)
    return


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class convnet(nn.Module):
    def __init__(self, num_classes=10):
        super(convnet, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x)  # 14x14

        x = self.conv2(x)
        x = self.relu(x)  # 14x14
        feat_out = x
        x = self.conv3(x)
        x = self.relu(x)  # 7x7
        x = self.conv4(x)
        x = self.relu(x)  # 7x7

        feat_low = x
        feat_low = self.avgpool(feat_low)
        feat_low = feat_low.view(feat_low.size(0), -1)
        y_low = self.fc(feat_low)

        return feat_out, y_low


class Predictor(nn.Module):
    def __init__(self, input_ch=32, num_classes=8):
        super(Predictor, self).__init__()
        self.pred_conv1 = nn.Conv2d(input_ch, input_ch, kernel_size=3, stride=1, padding=1)
        self.pred_bn1 = nn.BatchNorm2d(input_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pred_conv2 = nn.Conv2d(input_ch, num_classes, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        px = self.softmax(x)

        return x, px


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * 0.1


def grad_reverse(x):
    return GradReverse.apply(x)


class Trainer:
    def __init__(self, option):
        self.option = option
        self.option.cuda = torch.cuda.is_available() and not option.gpu < 0

        # ================================================================
        # ======================= build model ============================
        self.n_color_cls = 8

        self.option.n_class = 2 if self.option.dataset == "adult" else 10

        self.net = convnet(num_classes=self.option.n_class)
        self.pred_net_r = Predictor(input_ch=32, num_classes=self.n_color_cls)
        self.pred_net_g = Predictor(input_ch=32, num_classes=self.n_color_cls)
        self.pred_net_b = Predictor(input_ch=32, num_classes=self.n_color_cls)

        self.loss = nn.CrossEntropyLoss(ignore_index=255)
        self.color_loss = nn.CrossEntropyLoss(ignore_index=255)

        if self.option.cuda:
            self.net.cuda()
            self.pred_net_r.cuda()
            self.pred_net_g.cuda()
            self.pred_net_b.cuda()
            self.loss.cuda()
            self.color_loss.cuda()

        # =================================================================
        # ======================= set optimizer ===========================
        self.option.momentum = 0.9
        self.optim = optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.option.lr,
            momentum=self.option.momentum,
            weight_decay=self.option.weight_decay,
        )
        self.optim_r = optim.SGD(
            self.pred_net_r.parameters(),
            lr=self.option.lr,
            momentum=self.option.momentum,
            weight_decay=self.option.weight_decay,
        )
        self.optim_g = optim.SGD(
            self.pred_net_g.parameters(),
            lr=self.option.lr,
            momentum=self.option.momentum,
            weight_decay=self.option.weight_decay,
        )
        self.optim_b = optim.SGD(
            self.pred_net_b.parameters(),
            lr=self.option.lr,
            momentum=self.option.momentum,
            weight_decay=self.option.weight_decay,
        )

        # TODO: last_epoch should be the last step of loaded model
        self.option.lr_decay_rate = 0.1
        self.option.lr_decay_period = 40

        def _lr_lambda(step):
            return self.option.lr_decay_rate ** (step // self.option.lr_decay_period)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=_lr_lambda, last_epoch=-1
        )
        self.scheduler_r = optim.lr_scheduler.LambdaLR(
            self.optim_r, lr_lambda=_lr_lambda, last_epoch=-1
        )
        self.scheduler_g = optim.lr_scheduler.LambdaLR(
            self.optim_g, lr_lambda=_lr_lambda, last_epoch=-1
        )
        self.scheduler_b = optim.lr_scheduler.LambdaLR(
            self.optim_b, lr_lambda=_lr_lambda, last_epoch=-1
        )

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def _initialization(self):
        self.net.apply(self._weights_init)

        if self.option.is_train and self.option.use_pretrain:
            if self.option.checkpoint is not None:
                self._load_model()
            else:
                print("Pre-trained model not provided")

    def _mode_setting(self, is_train=True):
        if is_train:
            self.net.train()
            self.pred_net_r.train()
            self.pred_net_g.train()
            self.pred_net_b.train()
        else:
            self.net.eval()
            self.pred_net_r.eval()
            self.pred_net_g.eval()
            self.pred_net_b.eval()

    def _get_color_labels(self, images: torch.Tensor):
        label_image = F.interpolate(images, (14, 14))

        label_image = label_image * 255
        label_image = label_image.to(dtype=torch.int8)
        label_image /= 255

        return label_image

    def _train_step(self, data_loader, step):
        _lambda = 0.01
        start_time = time.monotonic()

        for i, (images, color_labels, labels) in enumerate(data_loader):
            if not self.option.use_ln2l_data:
                color_labels = self._get_color_labels(images)

            images = self._maybe_to_cuda(images)
            color_labels = self._maybe_to_cuda(color_labels)
            labels = self._maybe_to_cuda(labels)

            self.optim.zero_grad()
            self.optim_r.zero_grad()
            self.optim_g.zero_grad()
            self.optim_b.zero_grad()
            feat_label, pred_label = self.net(images)

            # predict colors from feat_label. Their prediction should be uniform.
            _, pseudo_pred_r = self.pred_net_r(feat_label)
            _, pseudo_pred_g = self.pred_net_g(feat_label)
            _, pseudo_pred_b = self.pred_net_b(feat_label)

            # loss for self.net
            loss_pred = self.loss(pred_label, torch.squeeze(labels))

            loss_pseudo_pred_r = torch.mean(torch.sum(pseudo_pred_r * torch.log(pseudo_pred_r), 1))
            loss_pseudo_pred_g = torch.mean(torch.sum(pseudo_pred_g * torch.log(pseudo_pred_g), 1))
            loss_pseudo_pred_b = torch.mean(torch.sum(pseudo_pred_b * torch.log(pseudo_pred_b), 1))

            loss_pred_ps_color = (
                loss_pseudo_pred_r + loss_pseudo_pred_g + loss_pseudo_pred_b
            ) / 3.0

            loss = loss_pred + loss_pred_ps_color * _lambda

            loss.backward()
            self.optim.step()

            self.optim.zero_grad()
            self.optim_r.zero_grad()
            self.optim_g.zero_grad()
            self.optim_b.zero_grad()

            feat_label, pred_label = self.net(images)

            feat_color = grad_reverse(feat_label)
            pred_r, _ = self.pred_net_r(feat_color)
            pred_g, _ = self.pred_net_g(feat_color)
            pred_b, _ = self.pred_net_b(feat_color)

            # loss for rgb predictors
            loss_pred_r = self.color_loss(pred_r, color_labels[:, 0])
            loss_pred_g = self.color_loss(pred_g, color_labels[:, 1])
            loss_pred_b = self.color_loss(pred_b, color_labels[:, 2])

            loss_pred_color = loss_pred_r + loss_pred_g + loss_pred_b

            loss_pred_color.backward()
            self.optim.step()
            self.optim_r.step()
            self.optim_g.step()
            self.optim_b.step()

            if i % self.option.log_step == 0:
                msg = "[TRAIN] cls loss : %.6f, rgb : %.6f, MI : %.6f  (epoch %d.%02d)" % (
                    loss_pred,
                    loss_pred_color / 3.0,
                    loss_pred_ps_color,
                    step,
                    int(100 * i / len(data_loader)),
                )
                print(msg)
        elapsed = time.monotonic() - start_time
        print(
            f"[TRAIN] Epoch {step} done. Elapsed time: {elapsed:.1f}s. "
            f"Batches per second: {len(data_loader) / elapsed:.1f}"
        )

    def _train_step_baseline(self, data_loader, step):
        start_time = time.monotonic()
        for i, (images, _, labels) in enumerate(data_loader):

            images = self._maybe_to_cuda(images)
            labels = self._maybe_to_cuda(labels)

            self.optim.zero_grad()
            feat_label, pred_label = self.net(images)

            # loss for self.net
            loss_pred = self.loss(pred_label, torch.squeeze(labels))
            loss_pred.backward()
            self.optim.step()

            if i % self.option.log_step == 0:
                msg = "[TRAIN] cls loss : %.6f (epoch %d.%02d)" % (
                    loss_pred,
                    step,
                    int(100 * i / len(data_loader)),
                )
                print(msg)
        elapsed = time.monotonic() - start_time
        print(
            f"[TRAIN] Epoch {step} done. Elapsed time: {elapsed:.1f}s. "
            f"Batches per second: {len(data_loader) / elapsed:.1f}"
        )

    def _validate(self, data_loader):
        self._mode_setting(is_train=False)
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()
        else:
            print("No trained model for evaluation provided")
            import sys

            sys.exit()

        num_test = 10000

        total_num_correct = 0.0
        total_num_test = 0.0
        total_loss = 0.0
        for i, (images, color_labels, labels) in enumerate(data_loader):
            if not self.option.use_ln2l_data:
                color_labels = self._get_color_labels(images)

            images = self._maybe_to_cuda(images)
            color_labels = self._maybe_to_cuda(color_labels)
            labels = self._maybe_to_cuda(labels)

            self.optim.zero_grad()
            _, pred_label = self.net(images)

            loss = self.loss(pred_label, torch.squeeze(labels))

            batch_size = images.shape[0]
            total_num_correct += self._num_correct(pred_label, labels, topk=1).item()
            total_loss += loss.item() * batch_size
            total_num_test += batch_size

        avg_loss = total_loss / total_num_test
        avg_acc = total_num_correct / total_num_test
        msg = "EVALUATION LOSS  %.4f, ACCURACY : %.4f (%d/%d)" % (
            avg_loss,
            avg_acc,
            int(total_num_correct),
            total_num_test,
        )
        print(msg)

    def _num_correct(self, outputs, labels, topk=1):
        _, preds = outputs.topk(k=topk, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).sum()
        return correct

    def _accuracy(self, outputs, labels):
        batch_size = labels.size(0)
        _, preds = outputs.topk(k=1, dim=1)
        preds = preds.t()
        correct = preds.eq(labels.view(1, -1).expand_as(preds))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        accuracy = correct.mul_(100.0 / batch_size)
        return accuracy

    def _save_model(self, step):
        torch.save(
            {
                "step": step,
                "optim_state_dict": self.optim.state_dict(),
                "net_state_dict": self.net.state_dict(),
            },
            os.path.join(
                self.option.save_dir, self.option.exp_name, "checkpoint_step_%04d.pth" % step
            ),
        )
        print("checkpoint saved. step : %d" % step)

    def _load_model(self):
        ckpt = torch.load(self.option.checkpoint)
        self.net.load_state_dict(ckpt["net_state_dict"])
        self.optim.load_state_dict(ckpt["optim_state_dict"])

    def train(self, train_loader, val_loader=None):
        self._initialization()
        if self.option.checkpoint is not None:
            self._load_model()

        self._mode_setting(is_train=True)
        start_epoch = 0
        for step in range(start_epoch, self.option.max_step):
            if self.option.train_baseline:
                self._train_step_baseline(train_loader, step)
            else:
                self._train_step(train_loader, step)
            self.scheduler.step()
            self.scheduler_r.step()
            self.scheduler_g.step()
            self.scheduler_b.step()

            if step == 1 or step % self.option.save_step == 0 or step == (self.option.max_step - 1):
                if val_loader is not None:
                    self._validate(val_loader)
                self._save_model(step)

    def _maybe_to_cuda(self, inputs):
        if self.option.cuda:
            return inputs.cuda()
        return inputs


class WholeDataLoader(Dataset):
    def __init__(self, option):
        self.data_split = option.data_split
        # data_dic = np.load(os.path.join(option.data_dir, 'mnist_10color_jitter_var_%.03f.npy' % option.scale),
        #                    encoding='latin1', allow_pickle=True).item()
        data_dic = np.load(
            Path(option.data_dir) / f"mnist_10color_jitter_var_{option.scale:.03f}.npz"
        )
        if self.data_split == "train":
            self.image = data_dic["train_image"]
            self.label = data_dic["train_label"]
        elif self.data_split == "test":
            self.image = data_dic["test_image"]
            self.label = data_dic["test_label"]

        color_var = option.scale
        self.color_std = color_var ** 0.5

        self.T = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

        self.ToPIL = transforms.Compose([transforms.ToPILImage()])

    def __getitem__(self, index):
        label = self.label[index]
        image = self.image[index]

        image = self.ToPIL(image)

        label_image = image.resize((14, 14), Image.NEAREST)

        label_image = torch.from_numpy(np.transpose(label_image, (2, 0, 1)))
        mask_image = torch.lt(label_image.float() - 0.00001, 0.0) * 255
        label_image = torch.div(label_image, 32)
        label_image = label_image + mask_image
        label_image = label_image.long()

        return self.T(image), label_image, label.astype(np.long)

    def __len__(self):
        return self.image.shape[0]


if __name__ == "__main__":
    main()
