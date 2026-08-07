import numpy as np

def compile_attention_mask(predicate, seq_len):
    q_indices = np.arange(seq_len)[:, None]
    kv_indices = np.arange(seq_len)[None, :]
    mask = predicate(q_indices, kv_indices)
    return mask
