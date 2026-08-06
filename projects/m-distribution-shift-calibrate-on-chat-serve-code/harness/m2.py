import ref
import numpy as np


def check(workdir):
    from calib.adjust import adjust_scales

    out = {"scales_adjusted": 0.0}
    try:
        shift = ref.get_oracle_shift()
        got_scales = adjust_scales(ref.CHAT_SCALES, shift)
        want_scales = ref.get_oracle_adjusted()
        if np.allclose(got_scales, want_scales, atol=1e-5):
            out["scales_adjusted"] = 1.0
        else:
            out["_note"] = "adjusted scales do not match expected reference scaling"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)}"
    return out
