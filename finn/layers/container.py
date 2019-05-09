import torch
import torch.nn as nn


class MultiHead(nn.Module):

    def __init__(self, head_list, split_dim):
        super(MultiHead, self).__init__()

        head_list = list(head_list)
        self.heads = nn.ModuleList(head_list)
        self.split_dim = split_dim

        assert len(head_list) == len(split_dim), "Number of heads must" \
                                                 " equal the number of specified splits"

        for i, head in enumerate(head_list):
            if not isinstance(head, SequentialFlow):
                head_list[i] = SequentialFlow(head_list[i])

    def forward(self, x, logpx=None, reverse=False, inds=None):

        xs = x.split(split_size=self.split_dim + [x.size(1) - sum(self.split_dim)], dim=1)
        outputs = []

        for x_, head in zip(xs, self.heads):
            if head is None:
                output_ = x_
            else:
                if logpx is not None:
                    output_, logpx_ = head(x_, logpx, reverse)
                    logpx += logpx_
                else:
                    output_ = head(x_, logpx, reverse)

            outputs.append(output_)

        outputs = torch.cat(outputs, dim=1)

        if logpx is None:
            return outputs
        else:
            return outputs, logpx


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layer_list):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layer_list)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx
