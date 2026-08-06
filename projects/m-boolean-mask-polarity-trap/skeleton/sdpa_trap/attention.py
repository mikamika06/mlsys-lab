import torch
import torch.nn.functional as F
import math

def compute_manual(q, k, v, pad_mask):
    """
    Compute standard multi-head attention.
    pad_mask: boolean tensor of shape (B, S) where True indicates a padding token.
    """
    raise NotImplementedError

def compute_sdpa(q, k, v, pad_mask):
    """
    Compute multi-head attention using F.scaled_dot_product_attention.
    pad_mask: boolean tensor of shape (B, S) where True indicates a padding token.
    """
    raise NotImplementedError
