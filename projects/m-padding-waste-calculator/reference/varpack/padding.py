"""Padding waste and efficiency calculations."""

import numpy as np


def compute_padding_waste(seq_lens, max_len=None):
    """Calculate ratio of padding tokens relative to total padded tokens."""
    lens = np.array(seq_lens, dtype=np.int64)
    if lens.size == 0:
        return 0.0
    target_len = int(max_len if max_len is not None else np.max(lens))
    total_padded = target_len * len(lens)
    if total_padded == 0:
        return 0.0
    total_real = int(np.sum(lens))
    return float((total_padded - total_real) / total_padded)


def compute_flop_savings(seq_lens):
    """Calculate relative FLOP savings when switching from square batch to varlen attention."""
    lens = np.array(seq_lens, dtype=np.int64)
    if lens.size == 0:
        return 0.0
    max_l = int(np.max(lens))
    padded_flops = len(lens) * (max_l ** 2)
    if padded_flops == 0:
        return 0.0
    varlen_flops = int(np.sum(lens ** 2))
    return float((padded_flops - varlen_flops) / padded_flops)
