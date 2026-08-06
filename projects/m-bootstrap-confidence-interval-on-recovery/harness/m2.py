import ref
import numpy as np

def check(workdir):
    from evalrep.bootstrap import bootstrap_recovery_ci

    out = {"ci_matched": 0.0, "rel_err": 1.0}
    base = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    quant = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    
    try:
        got = bootstrap_recovery_ci(base, quant, num_samples=500, alpha=0.05, seed=42)
        want = ref.compute_reference_ci(base, quant, num_samples=500, alpha=0.05, seed=42)
        
        got_vals = np.array([got.get("mean", 0), got.get("lower", 0), got.get("upper", 0)], dtype=float)
        want_vals = np.array([want["mean"], want["lower"], want["upper"]], dtype=float)
        
        rel_err = float(np.max(np.abs(got_vals - want_vals) / (np.abs(want_vals) + 1e-8)))
        out["rel_err"] = rel_err
        if rel_err <= 0.05:
            out["ci_matched"] = 1.0
        else:
            out["_note"] = f"CI values differ from reference: got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"bootstrap_recovery_ci failed: {type(e).__name__}: {str(e)[:120]}"
    return out
