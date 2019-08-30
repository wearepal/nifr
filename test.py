

import torch


x = torch.arange(start=0, end=24)
x = x.view(3, 4, 2)
shape = x.shape
x = x.view(x.size(0), 2, int(x.size(1) / 2), x.size(-1))
x = x.permute(0, 2, 1, 3).contiguous()
x = x.view(shape)
print(x.view(-1))

x = x.view(x.size(0), int(x.size(1) / 2), 2, x.size(-1))
x = x.permute(0, 2, 1, 3).contiguous()
x = x.view(-1)
print(x)
# print(x)
# x = x.view(2, -1).t().contiguous()
# x = x.view(-1)
# print(x)

