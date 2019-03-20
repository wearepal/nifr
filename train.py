"""Main training file"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pyro.distributions import MixtureOfDiagNormals
import pandas as pd

from utils import utils
from optimisation.custom_optimizers import Adam
import layers
from layers.adversarial import GradReverseDiscriminator


NDECS = 0
ARGS = None
LOGGER = None


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_x', metavar="PATH")
    parser.add_argument('--train_s', metavar="PATH")
    parser.add_argument('--train_y', metavar="PATH")
    parser.add_argument('--test_x', metavar="PATH")
    parser.add_argument('--test_s', metavar="PATH")
    parser.add_argument('--test_y', metavar="PATH")

    parser.add_argument('--train_new', metavar="PATH")
    parser.add_argument('--test_new', metavar="PATH")

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--dims', type=str, default="100-100")
    parser.add_argument('--nonlinearity', type=str, default="tanh")
    parser.add_argument('--glow', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)

    parser.add_argument('--early_stopping', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--disc_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save', type=str, default='experiments/cnf')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--val_freq', type=int, default=4)
    parser.add_argument('--log_freq', type=int, default=10)

    parser.add_argument('--zs_dim', type=int, default=2)
    parser.add_argument('-iw', '--independence_weight', type=float, default=1)
    parser.add_argument('--base_density', default='normal',
                        choices=['normal', 'dirichlet', 'binormal', 'logitbernoulli'])

    parser.add_argument('--jit', type=eval, default=False, choices=[True, False],
                        help='Should JIT compilation to static graph be used?')

    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use (if available)')

    return parser.parse_args()


def update_lr(optimizer, n_vals_without_improvement):
    global NDECS
    if NDECS == 0 and n_vals_without_improvement > ARGS.early_stopping // 3:
        for param_group in optimizer.param_groups:
            param_group["lr"] = ARGS.lr / 10
        NDECS = 1
    elif NDECS == 1 and n_vals_without_improvement > ARGS.early_stopping // 3 * 2:
        for param_group in optimizer.param_groups:
            param_group["lr"] = ARGS.lr / 100
        NDECS = 2
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = ARGS.lr / 10**NDECS


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load dataframe from a parquet file"""
    with path.open('rb') as f:
        df = pd.read_feather(f)
    return torch.tensor(df.values, dtype=torch.float32)


def load_data():
    """Load dataset from the files specified in ARGS and return it as PyTorch datasets"""
    train_x = load_dataframe(Path(ARGS.train_x))
    train_s = load_dataframe(Path(ARGS.train_s))
    train_y = load_dataframe(Path(ARGS.train_y))
    test_x = load_dataframe(Path(ARGS.test_x))
    test_s = load_dataframe(Path(ARGS.test_s))
    test_y = load_dataframe(Path(ARGS.test_y))
    return {'trn': TensorDataset(train_x, train_s, train_y),
            'val': TensorDataset(test_x, test_s, test_y)}, train_x.shape[1]


def convert_data(train_tuple, test_tuple):
    """
    Convert tuples of dataframes to pytorch datasets
    Args:
        train_tuple: tuple of dataframes with the training data
        test_tuple: tuple of dataframes with the test data

    Returns:
        a dictionary with the pytorch datasets
    """
    data = {'trn': TensorDataset(*[torch.tensor(df.values, dtype=torch.float32)
                                   for df in train_tuple]),
            'val': TensorDataset(*[torch.tensor(df.values, dtype=torch.float32)
                                   for df in test_tuple])}
    return data


def _build_model(input_dim):
    """Build the model with ARGS.depth many layers

    If ARGS.glow is true, then each layer includes 1x1 convolutions.
    """
    hidden_dims = tuple(map(int, ARGS.dims.split("-")))
    chain = []
    for i in range(ARGS.depth):
        if ARGS.glow:
            chain += [layers.BruteForceLayer(input_dim)]
        chain += [layers.MaskedCouplingLayer(input_dim, hidden_dims, 'alternate', swap=i % 2 == 0)]
        if ARGS.batch_norm:
            chain += [layers.MovingBatchNorm1d(input_dim, bn_lag=ARGS.bn_lag)]
    if ARGS.base_density == 'dirichlet2':
        chain.append(layers.SoftplusTransform())
    return layers.SequentialFlow(chain)


def compute_loss(x, s, model, discriminator, *, return_z=False):
    zero = x.new_zeros(x.size(0), 1)

    z, delta_logp = model(torch.cat([x, s], dim=1), zero)  # run model forward

    if ARGS.base_density == 'dirichlet':
        dist = torch.distributions.Dirichlet(z.new_ones(z.size(1)) / z.size(1))
        log_pz = dist.log_prob(z)
    elif ARGS.base_density == 'binormal':
        ones = z.new_ones(1, z.size(1))
        dist = MixtureOfDiagNormals(torch.cat([-ones, ones], 0), torch.cat([ones, ones], 0),
                                    z.new_ones(2))
        log_pz = dist.log_prob(z)
    elif ARGS.base_density == 'logitbernoulli':
        temperature = z.new_tensor(.5)
        prob_of_1 = 0.5 * z.new_ones(1, z.size(1))
        dist = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(temperature,
                                                                           probs=prob_of_1)
        log_pz = dist.log_prob(z.clamp(-100, 100))
        z = z.sigmoid()  # z is logits, so apply sigmoid before feeding to discriminator
    else:
        dist = torch.distributions.Normal(0, 1)
        log_pz = dist.log_prob(z).view(z.size(0), -1).sum(1, keepdim=True)  # log(p(z))

    # z_s0 = z[s[:, 0] == 0]
    # z_s1 = z[s[:, 0] == 1]

    # test_zs1 = torch.masked_select(z, s.byte()).view(-1, z.shape[1])
    # test_zs0 = torch.masked_select(z, ~s.byte()).view(-1, z.shape[1])

    # mmd = metrics.MMDStatistic(z_s0.size(0), z_s1.size(0))
    # mmd_loss = mmd(z_s0[:, :-ARGS.zs_dim], z_s1[:, :-ARGS.zs_dim], alphas=[1])
    zx = z[:, :-ARGS.zs_dim]
    # zs = z[:, -ARGS.zs_dim:]

    mmd_loss = F.binary_cross_entropy(discriminator(zx), s)
    mmd_loss *= ARGS.independence_weight

    log_px = log_pz - delta_logp
    loss = -torch.mean(log_px) + mmd_loss
    if return_z:
        return loss, z
    return loss, log_px.mean(), -mmd_loss


def restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model


def train(model, discriminator, optimizer, disc_optimizer, dataloader, experiment, epoch):
    model.train()
    
    loss_meter = utils.AverageMeter()
    time_meter = utils.AverageMeter()
    end = time.time()

    for itr, (x, s, y) in enumerate(dataloader, start=epoch * len(dataloader)):

        optimizer.zero_grad()
        disc_optimizer.zero_grad()

        x = cvt(x)
        s = cvt(s)
        loss, log_p_x, mmd_loss = compute_loss(x, s, model, discriminator, return_z=False)
        loss_meter.update(loss.item())

        loss.backward()
        optimizer.step()
        disc_optimizer.step()

        time_meter.update(time.time() - end)

        if itr % ARGS.log_freq == 0:
            # epoch = float(itr) / (len(trn) / float(ARGS.batch_size))
            LOGGER.info("Iter {:06d} | Time {:.4f}({:.4f}) | "
                        "Loss log_p_x: {:.6f} mmd_loss: {:.6f} ({:.6f}) | ", itr,
                        time_meter.val, time_meter.avg, log_p_x.item(), mmd_loss.item(),
                        loss_meter.avg)
            experiment.log_metric("Loss log_p_x", log_p_x.item(), step=itr)
            experiment.log_metric("Loss mmd", mmd_loss.item(), step=itr)
        itr += 1
        end = time.time()


def validate(model, discriminator, dataloader):
    model.eval()
    # start_time = time.time()
    with torch.no_grad():
        val_loss = utils.AverageMeter()
        for x_val, s_val, _ in dataloader:
            x_val = cvt(x_val)
            s_val = cvt(s_val)
            loss, log_p_x, mmd_loss = compute_loss(x_val, s_val, model, discriminator)

            val_loss.update(loss.item(), n=x_val.size(0))

    return val_loss.avg


def cvt(*tensors):
    """Put tensors on the correct device and set type to float32"""
    moved = [tensor.type(torch.float32).to(ARGS.device, non_blocking=True) for tensor in tensors]
    if len(moved) == 1:
        return moved[0]
    return tuple(moved)


def main(train_tuple=None, test_tuple=None, experiment=None):
    global ARGS, LOGGER

    ARGS = parse_arguments()

    experiment.log_multiple_params(vars(ARGS))

    test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    save_dir = Path(ARGS.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER = utils.get_logger(logpath=save_dir / 'logs', filepath=Path(__file__).resolve())
    LOGGER.info(ARGS)

    ARGS.device = torch.device(f"cuda:{ARGS.gpu}" if torch.cuda.is_available() else "cpu")

    LOGGER.info('Using {} GPUs.', torch.cuda.device_count())

    if train_tuple is None:
        data, n_dims = load_data()
    else:
        data = convert_data(train_tuple, test_tuple)
        n_dims = train_tuple.x.values.shape[1]

    train_loader = DataLoader(data['trn'], shuffle=True, batch_size=ARGS.batch_size)
    val_loader = DataLoader(data['val'], shuffle=False, batch_size=test_batch_size)
    # tst = DataLoader(data.tst, shuffle=False, batch_size=ARGS.test_batch_size)

    model = _build_model(n_dims + 1).to(ARGS.device)
    if ARGS.jit:
        sample_x = next(iter(train_loader))[0]
        sample_s = next(iter(train_loader))[1]
        zeros = sample_x.new_zeros(sample_x.size(0), 1)
        model = torch.jit.trace(model, (torch.cat([sample_x, sample_s], dim=1), zeros))
    discriminator = GradReverseDiscriminator([n_dims + 1 - ARGS.zs_dim]
                                             + [100, 100] + [1]).to(ARGS.device)

    if ARGS.resume is not None:
        checkpt = torch.load(ARGS.resume)
        model.load_state_dict(checkpt['state_dict'])

    LOGGER.info(model)
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(model))

    if not ARGS.evaluate:
        optimizer = Adam(model.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay)
        disc_optimizer = Adam(discriminator.parameters(), lr=ARGS.disc_lr)

        # time_meter = utils.RunningAverageMeter(0.98)
        # loss_meter = utils.RunningAverageMeter(0.98)

        best_loss = float('inf')

        n_vals_without_improvement = 0

        for epoch in range(ARGS.epochs):

            LOGGER.info('=====> Epoch {}', epoch)

            if n_vals_without_improvement > ARGS.early_stopping > 0:
                break

            with experiment.train():
                train(model, discriminator, optimizer, disc_optimizer, train_loader,
                      experiment, epoch)

            if epoch % ARGS.val_freq == 0:
                with experiment.test():
                    val_loss = validate(model, discriminator, val_loader)
                    experiment.log_metric("Loss", val_loss, step=(epoch + 1) * len(train_loader))

                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save({
                            'ARGS': ARGS,
                            'state_dict': model.state_dict(),
                        }, save_dir / 'checkpt.pth')
                        n_vals_without_improvement = 0
                    else:
                        n_vals_without_improvement += 1
                    update_lr(optimizer, n_vals_without_improvement)

                    log_message = (
                        '[VAL] Epoch {:06d} | Val Loss {:.6f} | '
                        'No improvement during validation: {:02d}/{:02d}'.format(
                            epoch, val_loss, n_vals_without_improvement, ARGS.early_stopping
                        )
                    )
                    LOGGER.info(log_message)

        LOGGER.info('Training has finished.')
        model_template = _build_model(n_dims + 1).to(ARGS.device)
        model = restore_model(model_template, save_dir / 'checkpt.pth').to(ARGS.device)

    LOGGER.info('Evaluating model on test set.')
    model.eval()
    test_encodings = encode_dataset(val_loader, model, LOGGER, cvt)
    # df_test.to_feather(ARGS.test_new)
    train_encodings = encode_dataset(
        DataLoader(data['trn'], shuffle=False, batch_size=test_batch_size), model, LOGGER, cvt)
    # df_train.to_feather(ARGS.train_new)
    return train_encodings, test_encodings


def encode_dataset(dataset, model, LOGGER, cvt):
    representation = []
    with torch.no_grad():
        # test_loss = utils.AverageMeter()
        for itr, (x, s, _) in enumerate(dataset):
            x = cvt(x)
            s = cvt(s)
            zero = x.new_zeros(x.size(0), 1)
            z, _ = model(torch.cat([x, s], dim=1), zero)
            if ARGS.base_density == 'logitbernoulli':
                z = z.sigmoid()

            # test_loss.update(loss.item(), n=x.shape[0])
            representation.append(z)
            LOGGER.info('Progress: {:.2f}%', itr / len(dataset) * 100)

    representation = torch.cat(representation, dim=0).cpu().detach().numpy()

    df = pd.DataFrame(representation)
    columns = df.columns.astype(str)
    df.columns = columns
    zx = df[columns[:-ARGS.zs_dim]]
    zs = df[columns[-ARGS.zs_dim:]]

    return df, zx, zs


if __name__ == '__main__':
    main()
