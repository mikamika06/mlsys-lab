import numpy as np

def standard_attention(q, k_contig, v_contig, context_lens):
    batch_size, num_heads, head_dim = q.shape
    out = np.zeros_like(q)
    scale = 1.0 / np.sqrt(head_dim)

    for b in range(batch_size):
        seq_len = context_lens[b]
        for h in range(num_heads):
            q_vec = q[b, h]
            k_mat = k_contig[b, :seq_len, h, :]
            v_mat = v_contig[b, :seq_len, h, :]

            scores = np.dot(k_mat, q_vec) * scale
            scores = scores - np.max(scores)
            probs = np.exp(scores)
            probs /= np.sum(probs)

            out[b, h] = np.dot(probs, v_mat)

    return out


def paged_attention(q, k_cache, v_cache, block_tables, context_lens):
    batch_size, num_heads, head_dim = q.shape
    num_blocks, block_size, _, _ = k_cache.shape
    out = np.zeros_like(q)
    scale = 1.0 / np.sqrt(head_dim)

    for b in range(batch_size):
        seq_len = context_lens[b]
        block_table = block_tables[b]

        for h in range(num_heads):
            q_vec = q[b, h]
            scores = []
            for i in range(seq_len):
                logical_block = i // block_size
                offset = i % block_size
                physical_block = block_table[logical_block]

                k_vec = k_cache[physical_block, offset, h, :]
                scores.append(np.dot(k_vec, q_vec) * scale)

            scores = np.array(scores)
            scores = scores - np.max(scores)
            probs = np.exp(scores)
            probs /= np.sum(probs)

            out_vec = np.zeros(head_dim)
            for i in range(seq_len):
                logical_block = i // block_size
                offset = i % block_size
                physical_block = block_table[logical_block]
                v_vec = v_cache[physical_block, offset, h, :]
                out_vec += probs[i] * v_vec

            out[b, h] = out_vec

    return out
