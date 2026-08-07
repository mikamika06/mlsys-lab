import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from varpack.padding import compute_padding_waste, compute_flop_savings

    out = {"waste_matches": 0.0}
    ok = True

    for ds in ref.DATASETS:
        want_w = ref.ref_compute_padding_waste(ds)
        got_w = compute_padding_waste(ds)
        if not np.isclose(want_w, got_w, rtol=1e-5, atol=1e-5):
            ok = False
            out["_note"] = f"Padding waste mismatch: got {got_w}, want {want_w}"
            break

        want_fs = ref.ref_compute_flop_savings(ds)
        got_fs = compute_flop_savings(ds)
        if not np.isclose(want_fs, got_fs, rtol=1e-5, atol=1e-5):
            ok = False
            out["_note"] = f"FLOP savings mismatch: got {got_fs}, want {want_fs}"
            break

    if ok:
        out["waste_matches"] = 1.0

    return out
