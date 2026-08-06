import numpy as np
import ref


def check(workdir):
    from imatrix.compute import compute_imatrix
    from imatrix.merge import merge_imatrices

    _, batches = ref.get_dataset()
    shards = []
    for b in batches:
        im = ref.reference_compute(b["activations"])
        shards.append({"count": b["count"], "data": im})

    want = ref.reference_merge(shards)
    got = merge_imatrices(shards)

    if not isinstance(got, dict) or "data" not in got or "count" not in got:
        return {"rel_err": 1.0, "_note": "Invalid merged structure"}

    if got["count"] != want["count"]:
        return {"rel_err": 1.0, "_note": f"Total count mismatch: got {got['count']}, want {want['count']}"}

    errs = []
    for k in want["data"]:
        w_val = want["data"][k]
        g_val = got["data"][k]
        denom = np.linalg.norm(w_val)
        if denom == 0:
            denom = 1.0
        err = float(np.linalg.norm(g_val - w_val) / denom)
        errs.append(err)

    max_err = max(errs) if errs else 1.0
    return {"rel_err": max_err}
