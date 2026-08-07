import ref
import numpy as np

def check(workdir):
    from attencurve.rescale import rescale_block
    rm, rs = 2.0, 1.5
    bm, bs = 4.0, 2.5
    want = ref.rescale_block(rm, rs, bm, bs, use_old_max=True)
    got = rescale_block(rm, rs, bm, bs, use_old_max=True)
    match = 1.0 if np.allclose(got, want) else 0.0
    out = {"rescale_match": match}
    if match == 0.0:
        out["_note"] = f"got {got}, reference {want}"
    return out
