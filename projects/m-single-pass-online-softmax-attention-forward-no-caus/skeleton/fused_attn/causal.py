import numpy as np


def block_split_causal_attention_forward(Q, K, V, sm_scale, block_size=32):
    """
    Computes causal online softmax attention using off-diagonal and diagonal block splits.
    Q, K, V are numpy arrays of shape (B, H, N, D).
    """
    raise NotImplementedError
