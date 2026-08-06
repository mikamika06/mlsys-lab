import torch
import torch.nn.functional as F


def fused_layernorm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)
