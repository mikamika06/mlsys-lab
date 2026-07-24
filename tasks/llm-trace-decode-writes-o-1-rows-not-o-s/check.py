import numpy as np

from mlsys import scorers, probe

D = 8              # head dimension used for every case
LARGE_S = 512      # cache length at which the O(1)-vs-O(S) op count is probed


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _ref_step(k_cache, v_cache, q, k_new, v_new):
    """Real oracle: append the new row and attend q over the whole grown cache.

    Computed straight from NumPy — no reference values are hardcoded.
    """
    k_cache = np.asarray(k_cache, dtype=np.float64)
    v_cache = np.asarray(v_cache, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    K = np.concatenate([k_cache, np.asarray(k_new, dtype=np.float64).reshape(1, -1)], axis=0)
    V = np.concatenate([v_cache, np.asarray(v_new, dtype=np.float64).reshape(1, -1)], axis=0)
    scores = (q @ K.T) / np.sqrt(q.shape[0])
    out = _softmax(scores) @ V
    return out, K, V


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    # ---- correctness vs the NumPy oracle over several cache lengths ----------
    max_err = 0.0
    for S in (0, 1, 5, 37):
        k_cache = rng.standard_normal((S, D))
        v_cache = rng.standard_normal((S, D))
        q = rng.standard_normal(D)
        k_new = rng.standard_normal(D)
        v_new = rng.standard_normal(D)

        out_ref, K_ref, V_ref = _ref_step(k_cache, v_cache, q, k_new, v_new)

        try:
            out, K2, V2 = sol.decode_step(
                k_cache.copy(), v_cache.copy(), q.copy(), k_new.copy(), v_new.copy()
            )
            out = np.asarray(out, dtype=np.float64)
            K2 = np.asarray(K2, dtype=np.float64)
            V2 = np.asarray(V2, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "op_count": float("inf")}

        # the "grew by exactly one row" contract: shapes must match the oracle
        if out.shape != out_ref.shape or K2.shape != K_ref.shape or V2.shape != V_ref.shape:
            return {"max_abs_err": float("inf"), "op_count": float("inf")}

        err = max(
            scorers.max_abs_err(out, out_ref),
            scorers.max_abs_err(K2, K_ref),
            scorers.max_abs_err(V2, V_ref),
        )
        if err > max_err:
            max_err = err

    # ---- measure: one decode step at a large cache must be O(1) Python work --
    k_big = rng.standard_normal((LARGE_S, D))
    v_big = rng.standard_normal((LARGE_S, D))
    q = rng.standard_normal(D)
    k_new = rng.standard_normal(D)
    v_new = rng.standard_normal(D)
    try:
        op_count = float(probe.count_line_events(sol.decode_step, k_big, v_big, q, k_new, v_new))
    except Exception:
        op_count = float("inf")

    return {"max_abs_err": float(max_err), "op_count": op_count}
