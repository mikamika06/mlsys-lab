import ref
import numpy as np


def check(workdir):
    from redscale.nonassoc import quantify_non_associativity
    out = {"mse_match": 0.0}
    try:
        ok = 0
        for arr in ref.TEST_ARRAYS:
            got = quantify_non_associativity(arr)
            want = ref.quantify_non_associativity(arr)
            if np.isclose(got, want, rtol=1e-5, atol=1e-5):
                ok += 1
        out["mse_match"] = 1.0 if ok == len(ref.TEST_ARRAYS) else 0.0
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:120]}"
    return out
