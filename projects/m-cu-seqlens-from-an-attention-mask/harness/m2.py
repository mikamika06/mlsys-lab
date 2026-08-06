import ref
import numpy as np


def check(workdir):
    from varlen.packing import unpad, pad
    cases = ref.generate_cases()
    ok = 0
    out = {"roundtrip_matched": 0.0}
    for i, mask in enumerate(cases):
        hidden = np.random.randn(mask.shape[0], mask.shape[1], 4)
        unp = unpad(hidden, mask)
        rep = pad(unp, mask)
        want = ref.ref_pad(ref.ref_unpad(hidden, mask), mask)
        if unp is not None and rep is not None and np.allclose(rep, want):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i} roundtrip failed"
    out["roundtrip_matched"] = 1.0 if ok == len(cases) else 0.0
    return out
