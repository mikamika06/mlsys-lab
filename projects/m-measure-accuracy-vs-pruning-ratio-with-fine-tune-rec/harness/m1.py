import ref
import numpy as np

def check(workdir):
    from prune_eval.metrics import compute_unrecovered_curve
    ratios = ref.RATIOS
    want = ref.compute_unrecovered_curve(ratios)
    try:
        got = compute_unrecovered_curve(ratios)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"raised {type(e).__name__}: {e}"}
    if not isinstance(got, (list, np.ndarray)) or len(got) != len(want):
        return {"rel_err": 1.0, "_note": "invalid output format or length"}
    err = np.mean(np.abs(np.array(got) - np.array(want)) / (np.abs(np.array(want)) + 1e-8))
    return {"rel_err": float(err)}
