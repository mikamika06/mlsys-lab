import numpy as np
import ref


def check(workdir):
    from nf4.quant import get_nf4_quantiles
    out = {"quantiles_matched": 0.0}
    try:
        got = get_nf4_quantiles()
        want = ref.get_nf4_quantiles()
        if got is not None and len(got) == 16 and np.allclose(got, want, atol=1e-4):
            out["quantiles_matched"] = 1.0
        else:
            out["_note"] = f"quantiles do not match reference: got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)}"
    return out
