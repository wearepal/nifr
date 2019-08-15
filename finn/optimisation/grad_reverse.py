from torch.autograd import Function


class GradReverse(Function):
    """Gradient reversal layer"""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(features, lambda_=1.0):
    return GradReverse.apply(features, lambda_)
