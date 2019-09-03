

import torch


def _split(x, frac):
    split_point = round(x.size(0) * frac)
    return x.split(split_size=[x.size(0) - split_point, split_point], dim=0)


len_chain = 4
inds = range(len_chain)
splits = {1: 0.5, 2: 0.5, 3: 0.5}
xs = []


x = torch.arange(start=0, end=20)
for i in inds:
    if i in splits:
        x_removed, x = _split(x, splits[i])
        xs.append(x_removed)
xs.append(x)
x = torch.cat(xs, dim=0)

inds = range(len_chain - 1, -1, -1)
xs = {}
for block_ind, frac in splits.items():
    x_removed, x = _split(x, frac=frac)
    xs[block_ind] = x_removed

for i in inds:
    if i in xs:
        x = torch.cat([xs[i], x], dim=0)
