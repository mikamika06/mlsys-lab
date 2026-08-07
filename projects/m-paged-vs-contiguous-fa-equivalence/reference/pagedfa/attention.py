import numpy as np


def paged_attention(q, k_pool, v_pool, block_tables, seq_lens, block_size, scale=None):
    """Compute attention using paged physical block lookup."""
    batch_size, q_len, num_heads, head_dim = q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    out = np.zeros((batch_size, q_len, num_heads, head_dim), dtype=q.dtype)

    for b in range(batch_size):
        slen = seq_lens[b]
        num_blocks = (slen + block_size - 1) // block_size
        k_list = []
        v_list = []
        for blk_idx in range(num_blocks):
            phy_blk = block_tables[b, blk_idx]
            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, slen)
            valid_len = end_pos - start_pos
            k_list.append(k_pool[phy_blk, :valid_len])
            v_list.append(v_pool[phy_blk, :valid_len])

        k_b = np.concatenate(k_list, axis=0)
        v_b = np.concatenate(v_list, axis=0)

        q_b = q[b] * scale
        q_b_perm = np.swapaxes(q_b, 0, 1)
        k_b_perm = np.swapaxes(k_b, 0, 1)
        v_b_perm = np.swapaxes(v_b, 0, 1)

        scores = np.matmul(q_b_perm, k_b_perm.transpose(0, 2, 1))
        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        out_b = np.matmul(probs, v_b_perm)
        out[b] = np.swapaxes(out_b, 0, 1)

    return out
