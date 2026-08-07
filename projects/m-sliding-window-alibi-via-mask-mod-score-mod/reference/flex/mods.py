import numpy as np

def alibi_score_mod(score, h, q, kv, num_heads):
    """Applies ALiBi score modification."""
    slope = 2 ** (-8.0 * (h + 1) / num_heads)
    return score - slope * (q - kv)

def sliding_window_mask_mod(b, h, q, kv, window):
    """Applies a sliding window and causal mask."""
    return (q >= kv) & ((q - kv) < window)
