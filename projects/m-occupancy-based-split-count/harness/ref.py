import numpy as np
from splitkv.occupancy import compute_split_count, partition_kv_ranges
from splitkv.combine import combine_splits, split_kv_attention
from splitkv.cost import model_reduction_cost, find_optimal_splits

CONFIGS = [
    {"batch_size": 1, "num_heads": 8, "kv_len": 16384, "block_size": 128, "num_sms": 108, "target_waves": 1, "max_splits": 128},
    {"batch_size": 32, "num_heads": 8, "kv_len": 4096, "block_size": 128, "num_sms": 108, "target_waves": 1, "max_splits": 128},
    {"batch_size": 1, "num_heads": 1, "kv_len": 64, "block_size": 128, "num_sms": 108, "target_waves": 1, "max_splits": 128},
    {"batch_size": 2, "num_heads": 4, "kv_len": 8192, "block_size": 64, "num_sms": 80, "target_waves": 2, "max_splits": 64},
    {"batch_size": 1, "num_heads": 4, "kv_len": 1000, "block_size": 128, "num_sms": 108, "target_waves": 1, "max_splits": 128},
    {"batch_size": 1, "num_heads": 2, "kv_len": 32768, "block_size": 128, "num_sms": 108, "target_waves": 1, "max_splits": 16},
]


def naive_attention(q, k, v):
    q_is_3d = (q.ndim == 3)
    if q_is_3d:
        q = np.expand_dims(q, axis=2)
    d_k = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d_k)
    m = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - m)
    lse = np.sum(exp_scores, axis=-1, keepdims=True)
    attn = exp_scores / np.maximum(lse, 1e-20)
    out = np.matmul(attn, v)
    lse_val = np.squeeze(m, axis=-1) + np.log(np.squeeze(lse, axis=-1))
    if q_is_3d:
        out = np.squeeze(out, axis=2)
        lse_val = np.squeeze(lse_val, axis=2)
    return out, lse_val
