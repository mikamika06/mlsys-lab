import sys
import tracemalloc
import numpy as np


def _oracle(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


def _align(ref, got):
    out = [np.array(x, dtype=np.float64, copy=True) for x in got]
    for i in range(ref[1].shape[0]):
        score = float(
            np.sum(ref[0][:, i] * out[0][:, i])
            + np.sum(ref[2][i, :] * out[2][i, :])
        )
        if score < 0:
            out[0][:, i] *= -1
            out[2][i, :] *= -1
    return tuple(out)


def _factor_err(ref, got):
    got = _align(ref, got)
    return float(max(
        np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)
        for a, b in zip(ref, got)
    ))


def _peak_bytes(fn, *args):
    tracemalloc.start()
    try:
        fn(*args)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(123)
    cases = [
        (rng.normal(size=(80, 50)), 5),
        (rng.normal(size=(2500, 1800)), 50),
    ]

    err = 0.0
    extra = 0

    for A, k in cases:
        ref = _oracle(A, k)
        try:
            got = sol.rank_k_factors(A, k)
        except Exception:
            return {
                "factor_rel_err": float("inf"),
                "extra_alloc_bytes": float("inf"),
            }

        err = max(err, _factor_err(ref, got))

        oracle_mem = _peak_bytes(_oracle, A, k)
        sol_mem = _peak_bytes(sol.rank_k_factors, A, k)
        extra = max(extra, sol_mem - oracle_mem)

    return {
        "factor_rel_err": float(err),
        "extra_alloc_bytes": float(extra),
    }
