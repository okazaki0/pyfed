import torch.nn.functional as F
import torch


@torch.jit.script
def cross_entropy(pred, target):
    return F.cross_entropy(input=pred, target=target)


@torch.jit.script
def nll_loss(pred, target):
    return F.nll_loss(input=pred, target=target)


def get_serializable_loss(loss):
    lf = {"nll_loss": nll_loss, "cross_entropy": cross_entropy}
    return lf[loss]
