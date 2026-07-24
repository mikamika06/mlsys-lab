import numpy as np


def sharded_attention_heads(q, k, v, wo, num_ranks):
    b, h, s, d = q.shape
    out = np.zeros((b, s, wo.shape[1]), dtype=np.float64)
    heads_per_rank = h // num_ranks
    scale = np.sqrt(float(d))

    for rank in range(num_ranks):
        start = rank * heads_per_rank
        end = start + heads_per_rank
        partial = np.zeros_like(out)

        for head in range(start, end):
            scores = np.matmul(q[:, head], np.transpose(k[:, head], (0, 2, 1))) / scale
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            head_out = np.matmul(probs, v[:, head])

            row_start = head * d
            row_end = row_start + d
            partial += np.matmul(head_out, wo[row_start:row_end])

        out += partial

    return out
