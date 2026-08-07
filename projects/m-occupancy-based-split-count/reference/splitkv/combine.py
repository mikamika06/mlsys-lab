import numpy as np
from splitkv.occupancy import partition_kv_ranges


def combine_splits(partial_max, partial_lse, partial_out):
    global_max = np.max(partial_max, axis=-1, keepdims=True)
    alpha = np.exp(partial_max - global_max)
    weights = partial_lse * alpha
    global_lse_sum = np.sum(weights, axis=-1, keepdims=True)
    safe_denom = np.maximum(global_lse_sum, 1e-20)
    norm_weights = weights / safe_denom
    combined_out = np.sum(partial_out * norm_weights[..., None], axis=-2)
    combined_lse = np.squeeze(global_max, axis=-1) + np.log(np.squeeze(safe_denom, axis=-1))
    return combined_out, combined_lse


def split_kv_attention(q, k, v, split_count):
    q_is_3d = (q.ndim == 3)
    if q_is_3d:
        q = np.expand_dims(q, axis=2)

    b_sz, h_sz, q_len, d_k = q.shape
    kv_len = k.shape[2]

    ranges = partition_kv_ranges(kv_len, split_count)

    p_max_list = []
    p_lse_list = []
    p_out_list = []

    scale = 1.0 / np.sqrt(d_k)

    for start, end in ranges:
        k_s = k[:, :, start:end, :]
        v_s = v[:, :, start:end, :]
        scores = np.matmul(q, np.swapaxes(k_s, -1, -2)) * scale
        m_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - m_s)
        l_s = np.sum(exp_s, axis=-1, keepdims=True)
        safe_l_s = np.maximum(l_s, 1e-20)
        o_s = np.matmul(exp_s / safe_l_s, v_s)

        p_max_list.append(m_s)
        p_lse_list.append(l_s)
        p_out_list.append(o_s)

    p_max = np.concatenate(p_max_list, axis=-1)
    p_lse = np.concatenate(p_lse_list, axis=-1)
    p_out = np.stack(p_out_list, axis=-2)

    comb_out, comb_lse = combine_splits(p_max, p_lse, p_out)

    if q_is_3d:
        comb_out = np.squeeze(comb_out, axis=2)
        comb_lse = np.squeeze(comb_lse, axis=2)

    return comb_out, comb_lse
