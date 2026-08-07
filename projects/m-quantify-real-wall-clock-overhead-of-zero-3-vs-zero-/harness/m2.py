import ref
import numpy as np

def check(workdir):
    from zeroperf.metrics import compute_overhead, compute_rel_err
    z2, z3 = ref.LOGS
    ref_oh = ref.compute_overhead(z2, z3, warmup=10)
    try:
        user_oh = compute_overhead(z2, z3, warmup=10)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"compute_overhead raised {e}"}

    err = ref.compute_rel_err(user_oh, ref_oh)
    return {"rel_err": float(err)}
