import ref
import numpy as np


def check(workdir):
    from quant.derive import derive_symmetric, derive_affine

    out = {"params_matched": 0.0}
    matched = 0
    total = 0

    for w in ref.TEST_WEIGHTS_SYM:
        total += 1
        try:
            s_got, z_got = derive_symmetric(w, bits=4)
            s_want, z_want = ref.derive_symmetric(w, bits=4)
            if np.isclose(s_got, s_want, rtol=1e-5, atol=1e-5) and z_got == z_want:
                matched += 1
        except Exception:
            pass

    for w in ref.TEST_WEIGHTS_AFF:
        total += 1
        try:
            s_got, z_got = derive_affine(w, bits=4)
            s_want, z_want = ref.derive_affine(w, bits=4)
            if np.isclose(s_got, s_want, rtol=1e-5, atol=1e-5) and z_got == z_want:
                matched += 1
        except Exception:
            pass

    out["params_matched"] = float(matched)
    if matched < total:
        out["_note"] = f"Matched {matched} out of {total} parameter derivations correctly."
    return out
