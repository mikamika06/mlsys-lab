import numpy as np
from ringattn.softmax import online_update


def ring_attention(q, k, v, num_devices):
    batch, seq_q, dim = q.shape
    _, seq_k, _ = k.shape
    scale = 1.0 / np.sqrt(dim)

    m = np.full((batch, seq_q, 1), -np.inf, dtype=np.float32)
    l = np.zeros((batch, seq_q, 1), dtype=np.float32)
    o = np.zeros_like(q, dtype=np.float32)

    k_chunks = np.array_split(k, num_devices, axis=1)
    v_chunks = np.array_split(v, num_devices, axis=1)

    for d in range(num_devices):
        scores = np.matmul(q, np.swapaxes(k_chunks[d], -1, -2)) * scale
        m, l, o = online_update(m, l, o, scores, v_chunks[d])

    return o
