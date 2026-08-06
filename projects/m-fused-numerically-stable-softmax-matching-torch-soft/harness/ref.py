import torch
import torch.nn.functional as F


def ref_softmax(x):
    return torch.softmax(x, dim=-1)


def ref_overflow(x):
    e = torch.exp(x)
    return e / torch.sum(e, dim=-1, keepdim=True)


def ref_layernorm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)
