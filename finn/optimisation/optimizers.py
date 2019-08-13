import torch


class AdamExtGrad(torch.optim.Adam):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        super(AdamExtGrad, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    def parameters(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    yield p

    def _apply_gradients(self, grads):
        for grad, param in zip(grads, self.parameters()):
            param.grad = grad

    def accumulate_gradients(
        self, loss, retain_graph=False, create_graph=False, allow_unused=True
    ):
        if not isinstance(loss, torch.Tensor):
            raise ValueError("Loss must be a tensor.")
        else:
            if not loss.requires_grad:
                raise ValueError("Loss must have requires_grad set to True")

        grads = torch.autograd.grad(
            loss,
            self.parameters(),
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )
        self._apply_gradients(grads)

    def step(self, grads=None, closure=None):
        if grads is not None:
            self._apply_gradients(grads)
        super().step(closure)
