import sys
import ref
import numpy as np

def check(workdir):
    sys.path.insert(0, workdir)
    out = {"err_match": 0.0}
    try:
        from adapter_merge.numerical import quantization_error, forward_equivalence
        ok = 0
        total = len(ref.NUMERICAL_FIXTURES)
        for (w, a, b, scale, x) in ref.NUMERICAL_FIXTURES:
            w_err_want = ref.quantization_error(w, a, b, scale)
            w_err_got = quantization_error(w, a, b, scale)

            f_err_want = ref.forward_equivalence(x, w, a, b, scale)
            f_err_got = forward_equivalence(x, w, a, b, scale)

            if np.isclose(w_err_want, w_err_got, atol=1e-5) and np.isclose(f_err_want, f_err_got, atol=1e-5):
                ok += 1
        out["err_match"] = float(ok) / total
    except Exception as e:
        out["_note"] = f"Failed: {type(e).__name__}: {str(e)}"
    finally:
        sys.path.pop(0)
    return out
