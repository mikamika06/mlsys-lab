import torch
import torch.nn.functional as F


def fused_logits_forward_ref(x, w):
    return torch.matmul(x, w.t())


def fused_logits_forward_triton(x, w):
    return torch.matmul(x, w.t())
