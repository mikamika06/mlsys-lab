import numpy as np

# Fixed latencies for L1, L2, L3 and DRAM (cycles)
LATENCIES = np.array([1.0, 4.0, 12.0, 100.0], dtype=np.float64)

def _ref_amats(hit_rates: np.ndarray) -> np.ndarray:
    """Compute reference AMAT values for an array of hit‑rate rows."""
    h_l1, h_l2, h_l3 = hit_rates.T
    m1 = 1 - h_l1
    m2 = 1 - h_l2
    m3 = 1 - h_l3
    return (
        LATENCIES[0]
        + m1 * (LATENCIES[1] + m2 * (LATENCIES[2] + m3 * LATENCIES[3]))
    )

def _gen_hit_rates() -> np.ndarray:
    """Deterministically generate the hit‑rate test matrix."""
    return np.array([
        [0.98, 0.95, 0.90],
        [1.00, 1.00, 1.00],
        [0.70, 0.80, 0.85],
        [0.60, 0.65, 0.70],
        [0.90, 0.92, 0.93]
    ], dtype=np.float64)

def grade(sol, fx) -> dict:
    """Grader for compute_amat."""
    hit_rates = _gen_hit_rates()
    ref_vals = _ref_amats(hit_rates)
    errs = []
    for hr, ref in zip(hit_rates, ref_vals):
        try:
            out = sol.compute_amat(hr)
        except Exception:
            return {"rel_err": float("inf")}
        # Ensure scalar output
        out_val = float(np.asarray(out).item())
        err = abs(out_val - ref) / (abs(ref) + 1e-12)
        errs.append(err)
    max_rel_err = max(errs) if errs else 0.0
    return {"rel_err": max_rel_err}
