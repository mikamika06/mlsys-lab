import numpy as np

def generate_inputs(seed=42):
    rng = np.random.default_rng(seed)
    batch_size = 1
    seq_len = 64
    num_heads = 4
    head_dim = 32
    world_size = 4

    q = rng.standard_normal((batch_size, seq_len, num_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((batch_size, seq_len, num_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((batch_size, seq_len, num_heads, head_dim)).astype(np.float32)
    return q, k, v, world_size

def reference_ring_attention(q, k, v, world_size):
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

def reference_ulysses_reshuffle(x, world_size, rank, forward=True):
    b, s_p, h_p, d = x.shape
    if forward:
        s = s_p * world_size
        h = h_p * world_size
        x_reshaped = x.reshape(b, world_size, s_p // world_size, world_size, h_p, d)
        x_transposed = np.transpose(x_reshaped, (0, 2, 1, 3, 4, 5))
        x_flat = x_transposed.reshape(b, s // world_size, h, d)
        h_chunk_size = h // world_size
        return x_flat[:, :, rank * h_chunk_size : (rank + 1) * h_chunk_size, :]
    else:
        h = h_p * world_size
        h_chunk_size = h_p
        s_chunk_size = s_p
        x_gathered = np.zeros((b, s_chunk_size * world_size, h_chunk_size, d), dtype=x.dtype)
        for r in range(world_size):
            x_gathered[:, r * s_chunk_size : (r + 1) * s_chunk_size, :, :] = x
        return x_gathered

def reference_crossover(seq_len, hidden_size, world_size):
    ring_comm = 2 * (world_size - 1) * seq_len * hidden_size * 4
    ulysses_comm = 2 * (world_size - 1) / world_size * seq_len * hidden_size * 4
    return ring_comm, ulysses_comm
