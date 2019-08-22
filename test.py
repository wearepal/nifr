import torch
import torch.nn as nn

from finn.optimisation import RAdam

param = nn.Parameter(torch.randn(3, 5, 7, 11))
mask = nn.Parameter(torch.rand_like(param))
optim_param = RAdam([param])
optim_mask = RAdam([mask])

out = (param * 2) / (1 + param)
out.register_hook(lambda grad: grad * mask)
loss = torch.sum(out**2)
grad_param = torch.autograd.grad(loss, param, create_graph=True)[0]
# grad_param *= mask
optim_param.step()
grad = torch.autograd.grad(grad_param, mask, grad_outputs=torch.ones_like(grad_param))
print(grad)
