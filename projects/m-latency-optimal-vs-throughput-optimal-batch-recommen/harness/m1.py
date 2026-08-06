import ref
import numpy as np


def check(workdir):
    from batchopt.profile import compute_curves

    out = {"curves_matched": 0.0}
    ok = 0
    for i, p in enumerate(ref.PROFILES):
        want_lat, want_tp = ref.compute_curves(p)
        got_lat, got_tp = compute_curves(p)
        if np.allclose(got_lat, want_lat) and np.allclose(got_tp, want_tp):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"profile {i} mismatch: got lat {got_lat}, want {want_lat}"
    if ok == len(ref.PROFILES):
        out["curves_matched"] = 1.0
    return out
