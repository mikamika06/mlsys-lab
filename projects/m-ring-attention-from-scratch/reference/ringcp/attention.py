import numpy as np

def ring_attention_forward(q, k, v, block_size):
    seq_len, head_dim = q.shape
    num_blocks = (seq_len + block_size - 1) // block_size

    o = np.zeros_like(q)
    l = np.zeros((seq_len, 1))
    m = np.full((seq_len, 1), -np.inf)

    k_blocks = [k[i*block_size:(i+1)*block_size] for i in range(num_blocks)]
    v_blocks = [v[i*block_size:(i+1)*block_size] for i in range(num_blocks)]

    for step in range(num_blocks):
        for b_idx in range(num_blocks):
            target_k_idx = (b_idx - step) % num_blocks
            kb = k_blocks[target_k_idx]
            vb = v_blocks[target_k_idx]

            start_idx = b_idx * block_size
            end_idx = min(seq_len, (b_idx + 1) * block_size)
            qb = q[start_idx:end_idx]

            scores = np.matmul(qb, kb.T) / np.sqrt(head_dim)
            block_m = np.max(scores, axis=-1, keepdims=True)

            new_m = np.maximum(m[start_idx:end_idx], block_m)
            exp_scores = np.exp(scores - new_m)
            old_scale = np.exp(m[start_idx:end_idx] - new_m)

            l[start_idx:end_idx] = l[start_idx:end_idx] * old_scale + np.sum(exp_scores, axis=-1, keepdims=True)

            o[start_idx:end_idx] = o[start_idx:end_idx] * old_scale + np.matmul(exp_scores, vb)
            m[start_idx:end_idx] = new_m

    out = o / l
    return out
