"""Main training file"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from utils import utils
from optimisation.custom_optimizers import Adam
import layers

NDECS = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', metavar="PATH", required=True)
    parser.add_argument('--train_s', metavar="PATH", required=True)
    parser.add_argument('--train_y', metavar="PATH", required=True)
    parser.add_argument('--test_x', metavar="PATH", required=True)
    parser.add_argument('--test_s', metavar="PATH", required=True)
    parser.add_argument('--test_y', metavar="PATH", required=True)

    parser.add_argument('--predictions', metavar="PATH", required=True)

    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--dims', type=str, default="100-100")
    parser.add_argument('--nonlinearity', type=str, default="tanh")
    parser.add_argument('--glow', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)

    parser.add_argument('--early_stopping', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save', type=str, default='experiments/cnf')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--val_freq', type=int, default=200)
    parser.add_argument('--log_freq', type=int, default=10)
    return parser.parse_args()


def update_lr(optimizer, n_vals_without_improvement, args):
    global NDECS
    if NDECS == 0 and n_vals_without_improvement > args.early_stopping // 3:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 10
        NDECS = 1
    elif NDECS == 1 and n_vals_without_improvement > args.early_stopping // 3 * 2:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 100
        NDECS = 2
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 10**NDECS


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load dataframe from a parquet file"""
    with path.open('rb') as f:
        df = pd.read_feather(f)
    return torch.tensor(df.values, dtype=torch.float32)


def load_data(args):
    """Load dataset from the files specified in args and return it as PyTorch datasets"""
    train_x = load_dataframe(Path(args.train_x))
    train_s = load_dataframe(Path(args.train_s))
    train_y = load_dataframe(Path(args.train_y))
    test_x = load_dataframe(Path(args.test_x))
    test_s = load_dataframe(Path(args.test_s))
    test_y = load_dataframe(Path(args.test_y))
    return {'trn': TensorDataset(train_x, train_s, train_y),
            'val': TensorDataset(test_x, test_s, test_y)}, train_x.shape[1]


def _build_model(input_dim, args):
    """Build the model with args.depth many layers

    If args.glow is true, then each layer includes 1x1 convolutions.
    """
    hidden_dims = tuple(map(int, args.dims.split("-")))
    chain = []
    for i in range(args.depth):
        if args.glow:
            chain += [layers.BruteForceLayer(input_dim)]
        chain += [layers.MaskedCouplingLayer(input_dim, hidden_dims, 'alternate', swap=i % 2 == 0)]
        if args.batch_norm:
            chain += [layers.MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag)]
    return layers.SequentialFlow(chain)


def compute_loss(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = model(x, zero)  # run model forward

    logpz = utils.standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


def restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model


def main(args):
    test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size
    # logger
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = utils.get_logger(logpath=save_dir / 'logs' , filepath=Path(__file__).resolve())
    logger.info(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def cvt(x):
        return x.type(torch.float32).to(device, non_blocking=True)

    logger.info('Using {} GPUs.', torch.cuda.device_count())

    data, n_dims = load_data(args)
    trn = DataLoader(data['trn'], shuffle=True, batch_size=args.batch_size)
    val = DataLoader(data['val'], shuffle=False, batch_size=test_batch_size)
    # tst = DataLoader(data.tst, shuffle=False, batch_size=args.test_batch_size)

    model = _build_model(n_dims, args).to(device)

    if args.resume is not None:
        checkpt = torch.load(args.resume)
        model.load_state_dict(checkpt['state_dict'])

    logger.info(model)
    logger.info("Number of trainable parameters: {}", utils.count_parameters(model))

    if not args.evaluate:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        time_meter = utils.RunningAverageMeter(0.98)
        loss_meter = utils.RunningAverageMeter(0.98)

        best_loss = float('inf')
        itr = 0
        n_vals_without_improvement = 0
        end = time.time()
        model.train()
        while True:
            if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping:
                break

            for x, _, _ in trn:
                if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping:
                    break

                optimizer.zero_grad()

                x = cvt(x)
                loss = compute_loss(x, model)
                loss_meter.update(loss.item())

                loss.backward()
                optimizer.step()

                time_meter.update(time.time() - end)

                if itr % args.log_freq == 0:
                    epoch = float(itr) / (len(trn) / float(args.batch_size))
                    logger.info("Iter {:06d} | Epoch {:.2f} | Time {:.4f}({:.4f}) | "
                                "Loss {:.6f}({:.6f}) | ", itr, epoch, time_meter.val,
                                time_meter.avg, loss_meter.val, loss_meter.avg)
                itr += 1
                end = time.time()

                # Validation loop.
                if itr % args.val_freq == 0:
                    model.eval()
                    # start_time = time.time()
                    with torch.no_grad():
                        val_loss = utils.AverageMeter()
                        for x_val, _, _ in val:
                            x_val = cvt(x_val)
                            val_loss.update(compute_loss(x_val, model).item(), n=x_val.shape[0])

                        if val_loss.avg < best_loss:
                            best_loss = val_loss.avg
                            torch.save({
                                'args': args,
                                'state_dict': model.state_dict(),
                            }, save_dir / 'checkpt.pth')
                            n_vals_without_improvement = 0
                        else:
                            n_vals_without_improvement += 1
                        update_lr(optimizer, n_vals_without_improvement, args)

                        log_message = (
                            '[VAL] Iter {:06d} | Val Loss {:.6f} | '
                            'NoImproveEpochs {:02d}/{:02d}'.format(
                                itr, val_loss.avg, n_vals_without_improvement, args.early_stopping
                            )
                        )
                        logger.info(log_message)
                    model.train()

        logger.info('Training has finished.')
        model = restore_model(model, save_dir / 'checkpt.pth').to(device)

    logger.info('Evaluating model on test set.')
    model.eval()

    with torch.no_grad():
        test_loss = utils.AverageMeter()
        for itr, (x, _, _) in enumerate(val):
            x = cvt(x)
            test_loss.update(compute_loss(x, model).item(), n=x.shape[0])
            logger.info('Progress: {:.2f}%', itr / (len(val) / test_batch_size))
        log_message = f'[TEST] Iter {itr:06d} | Test Loss {test_loss.avg:.6f} '
        logger.info(log_message)


if __name__ == '__main__':
    main(parse_arguments())
