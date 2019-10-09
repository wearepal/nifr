import torchvision
from torch import autograd


def get_data_dim(data_loader):
    x, _, _ = next(iter(data_loader))
    x_dim = x.shape[1:]

    return x_dim


def log_images(experiment, image_batch, name, nsamples=64, nrows=8, monochrome=False, prefix=None):
    """Make a grid of the given images, save them in a file and log them with Comet"""
    prefix = "train_" if prefix is None else f"{prefix}_"
    images = image_batch[:nsamples]
    if monochrome:
        images = images.mean(dim=1, keepdim=True)
    # torchvision.utils.save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
    shw = torchvision.utils.make_grid(images, nrow=nrows).clamp(0, 1).cpu()
    experiment.log_image(torchvision.transforms.functional.to_pil_image(shw), name=prefix + name)



def contrastive_gradient_penalty(network, input, penalty_amount=1.0):
    """Contrastive gradient penalty.

    This is essentially the optimization introduced by Mescheder et al 2018.

    Args:
        network: Network to apply penalty through.
        input: Input or list of inputs for network.
        penalty_amount: Amount of penalty.

    Returns:
        torch.Tensor: gradient penalty optimization.

    """

    def _get_gradient(inp, output):
        gradient = autograd.grad(
            outputs=output,
            inputs=inp,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        return gradient

    if not isinstance(input, (list, tuple)):
        input = [input]

    input = [inp.detach() for inp in input]
    input = [inp.requires_grad_() for inp in input]

    with torch.set_grad_enabled(True):
        output = network(*input)[-1]
    gradient = _get_gradient(input, output)
    gradient = gradient.view(gradient.size()[0], -1)
    penalty = (gradient ** 2).sum(1).mean()

    return penalty * penalty_amount
