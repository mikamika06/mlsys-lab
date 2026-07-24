import numpy as np
from mlsys import scorers

def grade(sol, fx):
    """Compare chunked attention output & peak bytes against a full-attention oracle."""
    # ── test cases ──────────────────────────────────────────────────────────
    cases = []

    # fixed simple case
    Q0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    K0 = np.array([[1.0, 0.0], [1.0, 0.0]])
    V0 = np.array([[1.0], [2.0]])
    cases.append((Q0, K0, V0, 1))
    cases.append((Q0, K0, V0, 2))  # single chunk, peak = 2*2*8 = 32

    # random cases (seeded for determinism)
    rng = np.random.RandomState(2024)
    for seed in [2024, 2025, 2026]:
        rs = np.random.RandomState(seed)
        n_q = rs.randint(4, 15)
        n_k = rs.randint(3, 10)
        d = rs.randint(2, 12)
        d_v = rs.randint(2, 8)
        Q = rs.randn(n_q, d)
        K = rs.randn(n_k, d)
        V = rs.randn(n_k, d_v)
        # pick a reasonable chunk_size (not larger than n_q)
        chunk_size = max(1, min(n_q, rs.randint(2, max(4, n_q // 2 + 1))))
        cases.append((Q, K, V, chunk_size))

    max_error = 0.0
    peak_ok = 0
    total = len(cases)

    for Q, K, V, chunk_size in cases:
        # ── oracle: full attention ────────────────────────────────────────
        d = Q.shape[1]
        scale = 1.0 / np.sqrt(d)
        S = (Q @ K.T) * scale                     # (n_q, n_k)
        # softmax (row‑wise, numerically stable)
        S_max = np.max(S, axis=-1, keepdims=True)
        exp_S = np.exp(S - S_max)
        probs = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
        ref_out = probs @ V                       # (n_q, d_v)

        expected_bytes = chunk_size * K.shape[0] * 8

        # ── student ────────────────────────────────────────────────────────
        try:
            out, peak = sol.chunked_attention(np.copy(Q),
                                              np.copy(K),
                                              np.copy(V),
                                              chunk_size)
        except Exception:
            return {"max_abs_err": float("inf"), "peak_bytes_acc": 0.0}

        err = scorers.max_abs_err(ref_out, out)
        if err > max_error:
            max_error = err
        if peak == expected_bytes:
            peak_ok += 1

    return {
        "max_abs_err": max_error,
        "peak_bytes_acc": peak_ok / total,
    }
