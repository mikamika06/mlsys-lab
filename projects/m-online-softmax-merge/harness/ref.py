import numpy as np


def reference_merge_online_softmax(m_a, l_a, o_a, m_b, l_b, o_b):
    m_new = np.maximum(m_a, m_b)
    alpha = np.exp(m_a - m_new)
    beta = np.exp(m_b - m_new)
    l_new = alpha * l_a + beta * l_b
    o_new = (alpha[:, None] * l_a[:, None] * o_a + beta[:, None] * l_b[:, None] * o_b) / l_new[:, None]
    return m_new, l_new, o_new


def reference_chunked_attention(q, k, v, chunk_size=64):
    seq_len_k, d_k = k.shape
    d_v = v.shape[1]
    batch_size = q.shape[0]
    scale = 1.0 / np.sqrt(d_k)

    m_cum = np.full((batch_size,), -np.inf, dtype=np.float64)
    l_cum = np.zeros((batch_size,), dtype=np.float64)
    o_cum = np.zeros((batch_size, d_v), dtype=np.float64)

    for i in range(0, seq_len_k, chunk_size):
        k_chunk = k[i:i + chunk_size]
        v_chunk = v[i:i + chunk_size]

        scores = (q @ k_chunk.T) * scale
        m_b = np.max(scores, axis=-1)
        scores_shift = scores - m_b[:, None]
        exp_scores = np.exp(scores_shift)
        l_b = np.sum(exp_scores, axis=-1)
        o_b = (exp_scores @ v_chunk) / l_b[:, None]

        if i == 0:
            m_cum, l_cum, o_cum = m_b, l_b, o_b
        else:
            m_cum, l_cum, o_cum = reference_merge_online_softmax(m_cum, l_cum, o_cum, m_b, l_b, o_b)

    return o_cum


def reference_compute_rel_err(got, want):
    denom = np.maximum(np.abs(want), 1e-12)
    return float(np.max(np.abs(got - want) / denom))


def reference_verify_tolerance_bounds(got, want, rtol=1e-5, atol=1e-8):
    diff = np.abs(got - want)
    bound = atol + rtol * np.abs(want)
    passed = bool(np.all(diff <= bound))
    max_rel = reference_compute_rel_err(got, want)
    return {"passed": passed, "max_rel_err": max_rel}


def reference_analyze_error(query_dim, seqlens, chunk_size=64, seed=42):
    rng = np.random.default_rng(seed)
    results = {}
    d_k = query_dim
    d_v = query_dim

    for n in seqlens:
        q = rng.normal(0.0, 1.0, size=(16, d_k)).astype(np.float64)
        k = rng.normal(0.0, 1.0, size=(n, d_k)).astype(np.float64)
        v = rng.normal(0.0, 1.0, size=(n, d_v)).astype(np.float64)

        scale = 1.0 / np.sqrt(d_k)
        scores = (q @ k.T) * scale
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        exact_out = probs @ v

        chunk_out = reference_chunked_attention(q, k, v, chunk_size=chunk_size)
        err = reference_compute_rel_err(chunk_out, exact_out)
        results[n] = err

    return results
