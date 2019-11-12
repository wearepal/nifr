"""Main training file"""
import time
from pathlib import Path

import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from ethicml.evaluators import run_metrics
from ethicml.utility.data_structures import DataTuple
from ethicml.metrics import Accuracy, ProbPos, TPR

from nosinn.data import DatasetTriplet
from nosinn.models.configs.classifiers import linear_disciminator, mp_32x32_net, mp_64x64_net, fc_net
from nosinn.models.factory import build_discriminator
from nosinn.models.classifier import Classifier
from nosinn.models.base import ModelBase
from nosinn.utils import utils
from nosinn.optimisation.utils import get_data_dim
from nosinn.optimisation.loss import grad_reverse

__all__ = ["main"]

NDECS = 0
ARGS = None
LOGGER = None
INPUT_SHAPE = ()


def train(args, encoder, classifier, adversary, dataloader, epoch: int) -> int:
    encoder.train()
    classifier.eval()
    adversary.train()

    class_loss_meter = utils.AverageMeter()
    inv_loss_meter = utils.AverageMeter()

    time_meter = utils.AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time
    start_itr = epoch * len(dataloader)
    for itr, (x, s, y) in enumerate(dataloader, start=start_itr):

        x, s, y = to_device(x, s, y)

        feat_label = encoder(x)
        pred_y_loss, _ = classifier.routine(feat_label, y)

        pred_s_logits = adversary(feat_label)
        pred_s_probs = pred_s_logits.softmax(dim=1)
        entropy_loss = torch.sum(pred_s_probs * pred_s_probs.log(), dim=1).mean()

        loss_pred = pred_y_loss + args.entropy_weight * entropy_loss

        encoder.zero_grad()
        adversary.zero_grad()

        loss_pred.backward()
        encoder.step()

        feat_label = encoder(x)
        feat_label = grad_reverse(feat_label)

        pred_s_logits = adversary(feat_label)
        loss_pred_s = adversary.apply_criterion(pred_s_logits, s).mean()

        loss_inv = loss_pred_s

        encoder.zero_grad()
        adversary.zero_grad()

        loss_inv.backward()

        encoder.step()
        adversary.step()

        class_loss_meter.update(loss_pred.item())
        inv_loss_meter.update(loss_inv.item())
        time_meter.update(time.time() - end)


    time_for_epoch = time.time() - start_epoch_time
    LOGGER.info(
        "[TRN] Epoch {:04d} | Duration: {:.3g}s | Batches/s: {:.4g} | "
        "Loss Pred Y: {:.5g} | Loss Inv: {:.5g}",
        epoch,
        time_for_epoch,
        1 / time_meter.avg,
        class_loss_meter.avg,
        inv_loss_meter.avg,
    )
    return itr


def test(encoder, classifier, val_loader, itr):
    encoder.eval()
    classifier.eval()

    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()
    with torch.no_grad():
        for x_val, s_val, y_val in val_loader:

            x_val, s_val, y_val = to_device(x_val, s_val, y_val)

            feat_label = encoder(x_val)
            loss, acc = classifier.routine(feat_label, y_val)

            loss_meter.update(loss)
            acc_meter.update(acc)

    return loss_meter.avg, acc_meter.avg


def to_device(*tensors):
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS.device, non_blocking=True) for tensor in tensors]
    if len(moved) == 1:
        return moved[0]
    return tuple(moved)


def main(args, datasets):
    """Main function

    Args:
        args: commandline arguments
        datasets: a Dataset object
        metric_callback: a function that computes metrics

    Returns:
        the trained model
    """
    assert isinstance(datasets, DatasetTriplet)
    # ==== initialize globals ====
    global ARGS, LOGGER, INPUT_SHAPE
    ARGS = args

    save_dir = Path(ARGS.save) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = utils.get_logger(logpath=save_dir / "logs", filepath=Path(__file__).resolve())
    LOGGER.info(ARGS)
    LOGGER.info("Save directory: {}", save_dir.resolve())
    # ==== check GPU ====
    ARGS.device = torch.device(
        f"cuda:{ARGS.gpu}" if (torch.cuda.is_available() and not ARGS.gpu < 0) else "cpu"
    )
    LOGGER.info("{} GPUs available. Using GPU {}", torch.cuda.device_count(), ARGS.gpu)

    # ==== construct dataset ====
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    train_loader = DataLoader(datasets.task_train, shuffle=True, batch_size=ARGS.test_batch_size)
    test_loader = DataLoader(datasets.task, shuffle=False, batch_size=ARGS.test_batch_size)

    # ==== construct networks ====
    INPUT_SHAPE = get_data_dim(train_loader)

    optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}

    if args.dataset == "celeba":
        input_dim = INPUT_SHAPE[0]
        classifier = mp_64x64_net(input_dim, args.y_dim)
        encoder = classifier[:-1]
        classifier = classifier[-1:]
        adv_input_shape = classifier[0].weight.size(1)
    elif args.dataset == "cmnist":
        input_dim = INPUT_SHAPE[0]
        classifier = mp_32x32_net(input_dim, args.y_dim)
        encoder = classifier[:-1]
        classifier = classifier[-1:]
        adv_input_shape = classifier[0].weight.size(1)
    else:
        input_dim = (INPUT_SHAPE,)
        encoder = fc_net(input_dim, 35, hidden_dims=[35])
        classifier = nn.Linear(35, args.y_dim)
        adv_input_shape = 35

    encoder.to(ARGS.device)
    classifier.to(ARGS.device)

    encoder = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
    classifier = Classifier(
        classifier,
        num_classes=ARGS.s_dim if ARGS.s_dim > 1 else 2,
        optimizer_kwargs=optimizer_kwargs)
    # Initialise Discriminator
    adv_fn = linear_disciminator

    adv_optimizer_kwargs = {"lr": args.disc_lr}

    adversary_kwargs = {
        "hidden_channels": ARGS.disc_channels,
        "num_blocks": ARGS.disc_depth,
        "use_bn": False,
    }

    adversary = build_discriminator(
        args,
        (adv_input_shape,),
        frac_enc=1,
        model_fn=adv_fn,
        model_kwargs=adversary_kwargs,
        optimizer_kwargs=adv_optimizer_kwargs,
    )
    adversary.to(args.device)

    # Train INN for N epochs
    for epoch in range(ARGS.epochs):

        itr = train(args, encoder, classifier, adversary, train_loader, epoch)

        if epoch % ARGS.val_freq == 0 and epoch != 0:
            test_loss, test_acc = test(encoder, classifier, test_loader, itr)

            LOGGER.info(
                "[TEST] Epoch {:04d} | Test Loss {:.6f} | Test Acc {:.6f}",
                epoch,
                test_loss,
                test_acc,
            )

    LOGGER.info("Training has finished.")

    classifier = Classifier(
        nn.Sequential(encoder, classifier),
        num_classes=ARGS.s_dim if ARGS.s_dim > 1 else 2
    )
    preds, ground_truths, sens = classifier.predict_dataset(test_loader, device=args.device)
    preds = pd.DataFrame(preds, columns=["labels"])
    ground_truths = DataTuple(x=pd.DataFrame(sens, columns=['sens']), s=pd.DataFrame(sens, columns=['sens']), y=pd.DataFrame(ground_truths, columns=['labels']))

    full_name = f"{args.dataset}_5kims"
    if args.dataset == "cmnist":
        full_name += "_greyscale" if args.greyscale else "_color"
    elif args.dataset == "celeba":
        full_name += f"_{args.celeba_sens_attr}"
        full_name += f"_{args.celeba_target_attr}"
    full_name += f"_{args.entropy_weight}ew"
    full_name += f"_{str(args.epochs)}epochs"
    metrics = run_metrics(
        preds, ground_truths,
        metrics=[Accuracy()],
        per_sens_metrics=[ProbPos(), TPR()] if args.dataset != "cmnist" else []
    )
    print(f"Results for {full_name}:")
    print("\n".join(f"\t\t{key}: {value:.4f}" for key, value in metrics.items()))
    print()

    if args.save is not None:
        save_to_csv = Path(args.save)
        if not save_to_csv.exists():
            save_to_csv.mkdir(exist_ok=True)

        assert isinstance(save_to_csv, Path)
        results_path = save_to_csv / full_name
        if args.dataset == "cmnist":
            value_list = ",".join([str(args.scale)] + [str(v) for v in metrics.values()])
        else:
            value_list = ",".join([str(args.task_mixing_factor)] + [str(v) for v in metrics.values()])

        if results_path.is_file():
            with results_path.open("a") as f:
                f.write(value_list + "\n")
        else:
            with results_path.open("w") as f:
                if args.dataset == "cmnist":
                    f.write(",".join(["Scale"] + [str(k) for k in metrics.keys()]) + "\n")
                else:
                    f.write(",".join(["Mix_fact"] + [str(k) for k in metrics.keys()]) + "\n")
                f.write(value_list + "\n")

if __name__ == "__main__":
    main()
