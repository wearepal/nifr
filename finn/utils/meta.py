import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from finn import layers
from finn.models.disc_models import conv_classifier
from finn.utils.make_functional import make_functional


def add_gradients(grads, model):
    for grad, param in zip(grads, model.parameters()):
        param.grad = grad


@torch.jit.script
def _weight_update(weight, grad, lr, weight_decay):
    l2 = 2 * weight_decay * weight
    return weight - lr * (grad + l2)


def meta_update(loss, parameters, lr, weight_decay=0):
    grads = torch.autograd.grad(loss, parameters)   # create_graph=True)
    return (_weight_update(param, grad, lr, weight_decay)
            for param, grad in zip(grads, parameters))


def inner_meta_loop(args, model, loss, meta_train, meta_test, pred_s=False):

    if not isinstance(meta_train, DataLoader):
        meta_train = DataLoader(meta_train, batch_size=args.meta_batch_size, shuffle=True)

    if not isinstance(meta_test, DataLoader):
        meta_test = DataLoader(meta_test, batch_size=args.test_batch_size, shuffle=False)

    if args.dataset == 'cmnist':
        meta_clf = conv_classifier(args.zy_dim, args.s_dim, depth=2)
    else:
        meta_clf = nn.Linear(args.zy_dim, args.y_dim)

    meta_clf.to(args.device)

    clf_optimizer = Adam(meta_clf.parameters(), lr=args.fast_lr,
                         weight_decay=args.meta_weight_decay)

    fast_model = make_functional(model)
    fast_weights = model.parameters()

    fast_weights = meta_update(loss, fast_weights, args.fast_lr, args.weight_decay)

    loss_fn = F.nll_loss if args.dataset == 'cmnist' else F.binary_cross_entropy_with_logits

    meta_clf.train()

    for epoch in range(args.meta_epochs):
        for x, s, y in meta_train:
            if pred_s:
                target = s
            else:
                target = y

            if loss_fn == F.nll_loss:
                target = target.long()

            x = x.to(args.device)
            target = target.to(args.device)

            z = fast_model(x, params=list(fast_weights))[:, -args.zy_dim:]

            if pred_s:
                z = layers.grad_reverse(z, lambda_=args.pred_s_from_zy_weight)

            preds = meta_clf(z)
            loss = loss_fn(preds, target, reduction='mean')

            grads = torch.autograd.grad(loss, meta_clf.parameters(), retain_graph=True)
            add_gradients(grads, meta_clf)
            clf_optimizer.step()
            fast_weights = meta_update(loss, fast_weights, args.fast_lr,
                                       args.weight_decay)

    meta_clf.eval()

    meta_loss = torch.zeros(1).to(args.device)
    for x, s, y in meta_train:

        if pred_s:
            target = s
        else:
            target = y

        if loss_fn == F.nll_loss:
            target = target.long()

        x = x.to(args.device)
        target = target.to(args.device)

        z = fast_model(x, params=list(fast_weights))[:, -args.zy_dim:]

        if pred_s:
            z = layers.grad_reverse(z, lambda_=args.pred_s_from_zy_weight)

        preds = meta_clf(z)
        class_loss = loss_fn(preds, target, reduction='sum')

        meta_loss += class_loss.sum()

    meta_loss /= len(meta_test.dataset)
    meta_loss.backward(retain_graph=True)

    return meta_loss, fast_weights
