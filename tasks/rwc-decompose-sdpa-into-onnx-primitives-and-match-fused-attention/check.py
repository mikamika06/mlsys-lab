import numpy as np

# ── NumPy oracle: the fused SDPA reference ──────────────────────────────
def _stable_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def _oracle_sdpa(Q, K, V, mask=None, scale=None):
    d_k = Q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d_k)
    # Fused: one MatMul, one scale, one mask add, softmax, one MatMul
    scores = Q @ np.swapaxes(K, -2, -1) * scale
    if mask is not None:
        scores = scores + mask
    weights = _stable_softmax(scores, axis=-1)
    return weights @ V

# ── Grader ───────────────────────────────────────────────────────────────
def grade(sol, fx) -> dict:
    rng = np.random.default_rng(123)

    cases = []

    # 1. Square, no mask
    Q = rng.standard_normal((6, 8))
    K = rng.standard_normal((6, 8))
    V = rng.standard_normal((6, 4))
    cases.append((Q, K, V, None, None))

    # 2. Explicit additive mask (a few blocked positions)
    Q = rng.standard_normal((4, 16))
    K = rng.standard_normal((4, 16))
    V = rng.standard_normal((4, 8))
    m = np.zeros((4, 4))
    m[0, 2] = -np.inf
    m[1, 3] = -np.inf
    m[3, 0] = -np.inf
    cases.append((Q, K, V, m, None))

    # 3. Causal mask, Lq == Lk
    L = 5
    Q = rng.standard_normal((L, 8))
    K = rng.standard_normal((L, 8))
    V = rng.standard_normal((L, 4))
    causal = np.triu(np.full((L, L), -np.inf), k=1)
    cases.append((Q, K, V, causal, None))

    # 4. Causal mask, Lq != Lk
    Q = rng.standard_normal((6, 8))
    K = rng.standard_normal((4, 8))
    V = rng.standard_normal((4, 3))
    causal = np.triu(np.full((6, 4), -np.inf), k=1)
    cases.append((Q, K, V, causal, None))

    # 5. Custom scale
    Q = rng.standard_normal((3, 12))
    K = rng.standard_normal((3, 12))
    V = rng.standard_normal((3, 5))
    cases.append((Q, K, V, None, 0.25))

    # 6. Batched 3-D inputs with causal mask
    Q = rng.standard_normal((2, 4, 8))
    K = rng.standard_normal((2, 6, 8))
    V = rng.standard_normal((2, 6, 3))
    batch_causal = np.triu(np.full((4, 6), -np.inf), k=1)
    cases.append((Q, K, V, batch_causal, None))

    worst_err = 0.0
    for Q, K, V, mask, scale in cases:
        try:
            oracle_out = _oracle_sdpa(Q, K, V, mask, scale)
            student_out = sol.decompose_sdpa(Q, K, V, mask, scale)
            student_out = np.asarray(student_out, dtype=np.float64)
            err = float(np.max(np.abs(oracle_out - student_out)))
            worst_err = max(worst_err, err)
        except Exception:
            worst_err = float("inf")
            break

    return {"max_abs_err": worst_err}
