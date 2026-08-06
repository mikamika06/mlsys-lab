import io
import math
import numpy as np


def serialize_kv(K: np.ndarray, V: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, K=np.asarray(K, dtype=np.float64), V=np.asarray(V, dtype=np.float64))
    return buffer.getvalue()


def decode_from_kv(Q: np.ndarray, payload: bytes) -> np.ndarray:
    buffer = io.BytesIO(payload)
    data = np.load(buffer)
    K = np.asarray(data["K"], dtype=np.float64)
    V = np.asarray(data["V"], dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    M = Q.shape[0]
    D = Q.shape[1]
    N = K.shape[0]
    P = V.shape[1]
    sqrt_D = math.sqrt(D)

    logits = []
    for i in range(M):
        row = []
        for j in range(N):
            dot_prod = 0.0
            for k in range(D):
                dot_prod += Q[i, k] * K[j, k]
            row.append(dot_prod / sqrt_D)
        logits.append(row)

    logits_sub = []
    for i in range(M):
        row = logits[i]
        m_val = row[0]
        for val in row:
            if val > m_val:
                m_val = val
        sub_row = []
        for val in row:
            sub_row.append(val - m_val)
        logits_sub.append(sub_row)

    weights = []
    for i in range(M):
        row = logits_sub[i]
        exp_row = []
        for val in row:
            exp_row.append(math.exp(val))
        weights.append(exp_row)

    weights_norm = []
    for i in range(M):
        row = weights[i]
        s_val = 0.0
        for val in row:
            s_val += val
        norm_row = []
        for val in row:
            norm_row.append(val / s_val)
        weights_norm.append(norm_row)

    result = []
    for i in range(M):
        res_row = []
        for p in range(P):
            val = 0.0
            for j in range(N):
                val += weights_norm[i][j] * V[j, p]
            res_row.append(val)
        result.append(res_row)

    return np.array(result, dtype=np.float64)
