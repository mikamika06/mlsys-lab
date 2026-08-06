import numpy as np


def dequantize_nf4(qweight, absmax, codebook, block_size=64):
    """Dequantizes 4-bit indices back to float array using NF4 codebook and absmax scale factors."""
    raise NotImplementedError


def lora_nf4_forward(x, qweight, absmax, codebook, lora_a, lora_b, scaling, compute_dtype="float32", block_size=64):
    """Computes y = x @ W_dequant.T + scaling * (x @ lora_a.T @ lora_b.T)."""
    raise NotImplementedError
