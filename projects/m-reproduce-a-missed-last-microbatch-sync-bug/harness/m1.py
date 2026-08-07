import ref
import numpy as np


def check(workdir):
    from syncbug.sim import simulate_accumulation
    errs = []
    for n_mb in ref.TEST_CASES_SIM:
        want = ref.simulate_accumulation(n_mb, sync_last=True)
        try:
            got = simulate_accumulation(n_mb, sync_last=True)
        except Exception as e:
            return {"rel_err": 1.0, "_note": f"Exception: {type(e).__name__}: {e}"}
        want_val = want["final_grad_sum"]
        got_val = got.get("final_grad_sum", 0.0)
        rel = abs(got_val - want_val) / (abs(want_val) + 1e-8)
        errs.append(rel)
        if got.get("synced") != want["synced"]:
            errs.append(1.0)
    mean_err = float(np.mean(errs)) if errs else 1.0
    return {"rel_err": mean_err}
