import numpy as np


def compute_cu_seqlens(attention_mask):
    lengths = np.sum(attention_mask, axis=1)
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
