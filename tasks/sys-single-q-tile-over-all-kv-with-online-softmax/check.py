import numpy as np

from mlsys import probe


def _dense_reference(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


def _cases():
    rng = np.random.default_rng(0)
    specs = [(8, 64, 4, 16), (12, 200, 6, 25), (5, 97, 3, 11)]
    out = []
    for n_q, n_kv, d, bs in specs:
        Q = rng.standard_normal((n_q, d))
        K = rng.standard_normal((n_kv, d))
        V = rng.standard_normal((n_kv, d))
        out.append((Q, K, V, bs))
    return out


_FAIL = {"max_abs_err": float("inf"), "loop_ratio": 0.0}


def grade(sol, fx) -> dict:
    """
    max_abs_err: your output vs a dense float64 NumPy oracle over several
    seeded (Q, K, V, kv_block_size) cases.

    loop_ratio: runs the submission on the same (Q, K, V) with
    kv_block_size = n_kv (1 tile) and n_kv // 8 (8 tiles), counting
    Python-level line events with a settrace-based probe (after a warm-up
    call). A real per-tile loop makes the 8-tile run execute markedly more
    lines; a solution that ignores kv_block_size and computes everything
    densely in one shot gives a ratio near 1.0.
    """
    worst_err = 0.0
    for Q, K, V, bs in _cases():
        ref = _dense_reference(Q, K, V)
        try:
            got = sol.flash_forward_single_q_tile(Q.copy(), K.copy(), V.copy(), bs)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return dict(_FAIL)
        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return dict(_FAIL)
        worst_err = max(worst_err, float(np.max(np.abs(got - ref))))

    rng = np.random.default_rng(1)
    n_q, n_kv, d = 20, 240, 8
    Q = rng.standard_normal((n_q, d))
    K = rng.standard_normal((n_kv, d))
    V = rng.standard_normal((n_kv, d))
    try:
        sol.flash_forward_single_q_tile(Q, K, V, n_kv)  # warm-up, untracked
        e_single = probe.count_line_events(sol.flash_forward_single_q_tile, Q, K, V, n_kv)
        e_multi = probe.count_line_events(sol.flash_forward_single_q_tile, Q, K, V, n_kv // 8)
    except Exception:
        return dict(_FAIL)

    loop_ratio = float(e_multi) / float(max(e_single, 1))
    return {"max_abs_err": worst_err, "loop_ratio": loop_ratio}
