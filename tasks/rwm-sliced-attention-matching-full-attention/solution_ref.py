import numpy as np

def chunked_attention(Q, K, V, chunk_size):
    """Scaled dot‑product attention with query‑chunking to bound memory."""
    n_q, d = Q.shape
    n_k = K.shape[0]
    scale = 1.0 / np.sqrt(d)

    outputs = []
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        Q_chunk = Q[start:end]                     # (chunk, d)

        # score matrix for this chunk
        scores = (Q_chunk @ K.T) * scale            # (chunk, n_k)

        # softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # output for this chunk
        out_chunk = probs @ V                       # (chunk, d_v)
        outputs.append(out_chunk)

    output = np.vstack(outputs)                    # (n_q, d_v)
    peak_bytes = chunk_size * n_k * 8
    return output, peak_bytes
