import ref
import numpy as np


def check(workdir):
    from unfoldfix.rewrite import rewrite_unfold
    out = {"unfold_matched": 0.0}
    ok = 0
    for case in ref.UNFOLD_TESTS:
        x = case["x"]
        ks = case["kernel_size"]
        stride = case["stride"]
        pad = case["padding"]
        dil = case["dilation"]
        got = rewrite_unfold(x, ks, stride, pad, dil)
        want = ref.ref_rewrite_unfold(x, ks, stride, pad, dil)
        if got is not None and np.allclose(got, want, atol=1e-5):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"mismatch in unfold rewrite: got shape {getattr(got, 'shape', None)}, want shape {want.shape}"
    out["unfold_matched"] = float(ok)
    return out
