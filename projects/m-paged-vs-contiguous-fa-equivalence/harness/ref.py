import numpy as np

def generate_fixtures(seed=42):
    np.random.seed(seed)
    batch_size = 4
    num_heads = 2
    head_dim = 16
    block_size = 4
    num_blocks = 32

    q = np.random.randn(batch_size, num_heads, head_dim)
    k_cache = np.random.randn(num_blocks, block_size, num_heads, head_dim)
    v_cache = np.random.randn(num_blocks, block_size, num_heads, head_dim)

    context_lens = np.array([5, 8, 12, 15])
    max_blocks_per_seq = (15 + block_size - 1) // block_size

    block_tables = np.zeros((batch_size, max_blocks_per_seq), dtype=int)
    available_blocks = np.random.permutation(num_blocks)
    idx = 0
    for b in range(batch_size):
        req_blocks = (context_lens[b] + block_size - 1) // block_size
        for i in range(req_blocks):
            block_tables[b, i] = available_blocks[idx]
            idx += 1

    return q, k_cache, v_cache, block_tables, context_lens

def reconstruct_contiguous(k_cache, v_cache, block_tables, context_lens):
    batch_size = len(context_lens)
    max_seq_len = max(context_lens)
    _, block_size, num_heads, head_dim = k_cache.shape

    k_contig = np.zeros((batch_size, max_seq_len, num_heads, head_dim))
    v_contig = np.zeros((batch_size, max_seq_len, num_heads, head_dim))

    for b in range(batch_size):
        seq_len = context_lens[b]
        for i in range(seq_len):
            logical_block = i // block_size
            offset = i % block_size
            physical_block = block_tables[b][logical_block]
            k_contig[b, i] = k_cache[physical_block, offset]
            v_contig[b, i] = v_cache[physical_block, offset]

    return k_contig, v_contig

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
