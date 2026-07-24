import numpy as np


def batched_paged_decode(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    block_size,
):
    q = np.asarray(q, dtype=np.float64)
    k_cache = np.asarray(k_cache, dtype=np.float64)
    v_cache = np.asarray(v_cache, dtype=np.float64)
    block_tables = np.asarray(block_tables)
    seq_lens = np.asarray(seq_lens)

    batch, d = q.shape
    out = np.zeros((batch, d), dtype=np.float64)
    scale = np.sqrt(float(d))

    for b in range(batch):
        k_tokens = []
        v_tokens = []
        for t in range(int(seq_lens[b])):
            physical_block = int(block_tables[b, t // block_size])
            offset = t % block_size
            k_tokens.append(k_cache[physical_block, offset])
            v_tokens.append(v_cache[physical_block, offset])

        k = np.asarray(k_tokens, dtype=np.float64)
        v = np.asarray(v_tokens, dtype=np.float64)

        scores = (k @ q[b]) / scale
        scores -= np.max(scores)
        weights = np.exp(scores)
        weights /= np.sum(weights)
        out[b] = weights @ v

    return out
