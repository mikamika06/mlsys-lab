import numpy as np

from mlsys import scorers


def _cases():
    """Deterministic (flops, bytes, peak, bandwidth) machines/kernels."""
    rng = np.random.default_rng(0)
    cases = []
    for peak, bw in [(4e12, 2e11), (1.5e13, 1.0e12), (8e11, 5e10)]:
        ridge = peak / bw
        # intensities that straddle the ridge point on both sides
        ai = ridge * np.concatenate([
            rng.uniform(0.01, 0.9, 6),
            rng.uniform(1.1, 40.0, 6),
        ])
        bytes_moved = rng.uniform(1e7, 5e9, ai.size)
        flops = ai * bytes_moved
        cases.append((flops, bytes_moved, float(peak), float(bw)))
    return cases


def grade(sol, fx) -> dict:
    ai_err = 0.0
    att_err = 0.0
    ridge_err = 0.0

    for flops, bytes_moved, peak, bw in _cases():
        ref_ai = np.asarray(flops, dtype=np.float64) / np.asarray(bytes_moved, dtype=np.float64)
        ref_att = np.minimum(peak, ref_ai * bw)
        ref_ridge = peak / bw

        try:
            out = sol.roofline_attainable(flops.copy(), bytes_moved.copy(), peak, bw)
            got_ai, got_att, got_ridge = out
            got_ai = np.asarray(got_ai, dtype=np.float64).reshape(-1)
            got_att = np.asarray(got_att, dtype=np.float64).reshape(-1)
            got_ridge = float(np.asarray(got_ridge, dtype=np.float64).reshape(()))
        except Exception:
            return {
                "ai_rel_err": float("inf"),
                "attainable_rel_err": float("inf"),
                "ridge_rel_err": float("inf"),
            }

        if got_ai.shape != ref_ai.shape or got_att.shape != ref_att.shape:
            return {
                "ai_rel_err": float("inf"),
                "attainable_rel_err": float("inf"),
                "ridge_rel_err": float("inf"),
            }
        if not (np.all(np.isfinite(got_ai)) and np.all(np.isfinite(got_att)) and np.isfinite(got_ridge)):
            return {
                "ai_rel_err": float("inf"),
                "attainable_rel_err": float("inf"),
                "ridge_rel_err": float("inf"),
            }

        ai_err = max(ai_err, scorers.rel_err(ref_ai, got_ai))
        att_err = max(att_err, scorers.rel_err(ref_att, got_att))
        ridge_err = max(ridge_err, scorers.rel_err(np.array([ref_ridge]), np.array([got_ridge])))

    return {
        "ai_rel_err": float(ai_err),
        "attainable_rel_err": float(att_err),
        "ridge_rel_err": float(ridge_err),
    }
