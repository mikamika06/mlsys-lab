import math
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
    scale = math.sqrt(float(d))

    for b in range(batch):
        length = int(seq_lens[b])
        k_list = []
        v_list = []
        for t in range(length):
            logical_block = t // block_size
            offset = t % block_size
            physical_block = int(block_tables[b, logical_block])
            k_list.append(k_cache[physical_block, offset])
            v_list.append(v_cache[physical_block, offset])

        scores = []
        for t in range(length):
            dot_val = 0.0
            for j in range(d):
                dot_val += k_list[t][j] * q[b, j]
            scores.append(dot_val / scale)

        max_score = scores[0]
        for t in range(1, length):
            if scores[t] > max_score:
                max_score = scores[t]

        weights = []
        sum_weights = 0.0
        for t in range(length):
            w = math.exp(scores[t] - max_score)
            weights.append(w)
            sum_weights += w

        normalized_weights = []
        for t in range(length):
            normalized_weights.append(weights[t] / sum_weights)

        for j in range(d):
            val = 0.0
            for t in range(length):
                val += normalized_weights[t] * v_list[t][j]
            out[b, j] = val

    return out
