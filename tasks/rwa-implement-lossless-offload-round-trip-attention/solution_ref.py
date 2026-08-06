import numpy as np
import io
import math

def offload_attention(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    buf = io.BytesIO()
    np.savez_compressed(buf, k=k, v=v)
    buf.seek(0)
    data = np.load(buf, allow_pickle=False)
    k_off = data['k']
    v_off = data['v']

    B, N, D = q.shape
    _, S, _ = k_off.shape
    _, _, D_v = v_off.shape

    scale = math.sqrt(D)

    out = np.empty((B, N, D_v), dtype=np.float64)

    for b in range(B):
        for i in range(N):
            logits = []
            for j in range(S):
                dot = 0.0
                for d in range(D):
                    dot += q[b, i, d] * k_off[b, j, d]
                logits.append(dot / scale)

            max_val = logits[0]
            for j in range(1, S):
                if logits[j] > max_val:
                    max_val = logits[j]

            exp_logits = []
            exp_sum = 0.0
            for j in range(S):
                val = math.exp(logits[j] - max_val)
                exp_logits.append(val)
                exp_sum += val

            probs = []
            for j in range(S):
                probs.append(exp_logits[j] / exp_sum)

            for dv in range(D_v):
                val = 0.0
                for j in range(S):
                    val += probs[j] * v_off[b, j, dv]
                out[b, i, dv] = val

    return out.astype(np.float64)
