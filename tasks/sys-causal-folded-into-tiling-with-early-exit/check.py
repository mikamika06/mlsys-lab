import numpy as np


def _dense_causal(Q, K, V):
    S, d = Q.shape
    scores = (Q @ K.T) / np.sqrt(d)
    qi = np.arange(S)[:, None]
    kj = np.arange(S)[None, :]
    scores = np.where(kj > qi, -np.inf, scores)
    m = np.max(scores, axis=1, keepdims=True)
    e = np.exp(scores - m)
    probs = e / np.sum(e, axis=1, keepdims=True)
    return probs @ V


def _expected_visited(seq_len, tile_q, tile_kv):
    n_q = seq_len // tile_q
    n_kv = seq_len // tile_kv
    visited = set()
    for qi in range(n_q):
        q_end = (qi + 1) * tile_q - 1
        for kj in range(n_kv):
            k_start = kj * tile_kv
            if k_start > q_end:
                continue
            visited.add((qi, kj))
    return visited


def _build_cases():
    cases = []
    for seed, seq_len, d, tile_q, tile_kv in [
        (0, 24, 8, 8, 6),
        (1, 16, 4, 4, 4),
        (2, 20, 6, 5, 10),
        (3, 12, 5, 3, 4),
    ]:
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((seq_len, d))
        K = rng.standard_normal((seq_len, d))
        V = rng.standard_normal((seq_len, d))
        cases.append((Q, K, V, tile_q, tile_kv))
    return cases


def grade(sol, fx) -> dict:
    worst_err = 0.0
    visit_ok = 1.0

    for Q, K, V, tile_q, tile_kv in _build_cases():
        ref = _dense_causal(Q, K, V)
        expected_visited = _expected_visited(Q.shape[0], tile_q, tile_kv)

        visited = []
        try:
            got = sol.tiled_causal_attention(Q.copy(), K.copy(), V.copy(), tile_q, tile_kv,
                                              on_tile=lambda qi, kj: visited.append((qi, kj)))
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "tile_visit_exact_match": 0.0}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf"), "tile_visit_exact_match": 0.0}

        worst_err = max(worst_err, float(np.max(np.abs(got - ref))))

        if set(visited) != expected_visited or len(visited) != len(expected_visited):
            visit_ok = 0.0

    return {"max_abs_err": worst_err, "tile_visit_exact_match": visit_ok}
