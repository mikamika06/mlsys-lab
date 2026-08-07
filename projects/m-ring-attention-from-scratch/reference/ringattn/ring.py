import numpy as np

def ring_attention(q, k, v, world_size):
    b, s, h, d = q.shape
    chunk_size = s // world_size
    q_chunks = np.split(q, world_size, axis=1)
    k_chunks = np.split(k, world_size, axis=1)
    v_chunks = np.split(v, world_size, axis=1)

    outputs = []
    for rank in range(world_size):
        qr = q_chunks[rank]
        o_accum = np.zeros_like(qr)
        m_accum = np.full((b, h, qr.shape[1], 1), -np.inf, dtype=np.float32)
        l_accum = np.zeros((b, h, qr.shape[1], 1), dtype=np.float32)

        for step in range(world_size):
            kv_idx = (rank - step) % world_size
            kr = k_chunks[kv_idx]
            vr = v_chunks[kv_idx]

            scores = np.matmul(qr, np.swapaxes(kr, -1, -2)) / np.sqrt(d)
            m_block = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - m_block)
            l_block = np.sum(exp_scores, axis=-1, keepdims=True)

            m_new = np.maximum(m_accum, m_block)
            alpha = np.exp(m_accum - m_new)
            beta = np.exp(m_block - m_new)

            l_new = alpha * l_accum + beta * l_block

            o_accum = (alpha * l_accum * o_accum + beta * np.matmul(exp_scores, vr)) / l_new
            m_accum = m_new
            l_accum = l_new

        outputs.append(o_accum)
    return np.concatenate(outputs, axis=1)
