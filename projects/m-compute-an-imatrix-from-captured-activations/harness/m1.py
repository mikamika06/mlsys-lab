import numpy as np
import ref


def check(workdir):
    from imatrix.compute import compute_imatrix

    _, batches = ref.get_dataset()
    sample_acts = batches[0]["activations"]

    want = ref.reference_compute(sample_acts)
    got = compute_imatrix(sample_acts)

    if not isinstance(got, dict) or set(got.keys()) != set(want.keys()):
        return {"rel_err": 1.0, "_note": "Key mismatch or non-dict output"}

    errs = []
    for k in want:
        w_val = want[k]
        g_val = got[k]
        denom = np.linalg.norm(w_val)
        if denom == 0:
            denom = 1.0
        err = float(np.linalg.norm(g_val - w_val) / denom)
        errs.append(err)

    max_err = max(errs) if errs else 1.0
    return {"rel_err": max_err}
