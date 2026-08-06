import numpy as np
import math

def chunked_attention(Q, K, V, chunk_size):
    """Scaled dot‑product attention with query‑chunking to bound memory."""
    n_q, d = Q.shape
    n_k = K.shape[0]
    d_v = V.shape[1]
    scale = 1.0 / math.sqrt(d)

    outputs = []
    for start in range(0, n_q, chunk_size):
        end = min(start + chunk_size, n_q)
        Q_chunk = Q[start:end]
        chunk_len = end - start

        scores = []
        for i in range(chunk_len):
            row = []
            for j in range(n_k):
                dot = 0.0
                for k in range(d):
                    dot += Q_chunk[i, k] * K[j, k]
                row.append(dot * scale)
            scores.append(row)

        probs = []
        for i in range(chunk_len):
            row = scores[i]
            m = row[0]
            for val in row:
                if val > m:
                    m = val

            exp_row = []
            s_sum = 0.0
            for val in row:
                ex = math.exp(val - m)
                exp_row.append(ex)
                s_sum += ex

            prob_row = [ex / s_sum for ex in exp_row]
            probs.append(prob_row)

        out_chunk = []
        for i in range(chunk_len):
            out_row = []
            for j in range(d_v):
                val = 0.0
                for k in range(n_k):
                    val += probs[i][k] * V[k, j]
                out_row.append(val)
            out_chunk.append(out_row)

        outputs.append(np.array(out_chunk, dtype=Q.dtype))

    output = np.vstack(outputs)
    peak_bytes = chunk_size * n_k * 8
    return output, peak_bytes
