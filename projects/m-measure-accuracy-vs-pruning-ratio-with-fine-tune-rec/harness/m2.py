import ref
import numpy as np

def check(workdir):
    from prune_eval.finetune import evaluate_sweep
    ratios = ref.RATIOS
    want = ref.evaluate_sweep(ratios)
    try:
        got = evaluate_sweep(ratios)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"raised {type(e).__name__}: {e}"}
    if not isinstance(got, dict) or "unrecovered" not in got or "recovered" not in got:
        return {"rel_err": 1.0, "_note": "invalid output dict format"}
    want_arr = np.array(want["recovered"])
    got_arr = np.array(got["recovered"])
    err = np.mean(np.abs(got_arr - want_arr) / (np.abs(want_arr) + 1e-8))
    return {"rel_err": float(err)}
