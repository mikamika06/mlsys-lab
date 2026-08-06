import ref
import numpy as np


def check(workdir):
    from calib.shift import compute_shift
    from calib.metrics import relative_error

    out = {"rel_err_match": 0.0}
    try:
        got_shift = compute_shift(ref.CHAT_ACTS, ref.CODE_ACTS)
        want_shift = ref.get_oracle_shift()
        err = relative_error(want_shift, got_shift)
        if err < 1e-4:
            out["rel_err_match"] = 1.0
        else:
            out["_note"] = f"relative error between computed and oracle shift is {err}"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)}"
    return out
