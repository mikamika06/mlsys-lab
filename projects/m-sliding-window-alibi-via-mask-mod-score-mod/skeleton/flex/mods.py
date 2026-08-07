import numpy as np

def alibi_score_mod(score, h, q, kv, num_heads):
    """Applies ALiBi score modification."""
    raise NotImplementedError

def sliding_window_mask_mod(b, h, q, kv, window):
    """Applies a sliding window and causal mask."""
    raise NotImplementedError
