import ref
import numpy as np

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from qformat.dequant import dequantize_w4a16, dequantize_nvfp4

    out = {"max_abs_err_w4": 1e9, "max_abs_err_nv": 1e9}

    packed, scales, zeros, gs = ref.generate_w4a16_fixture()
    want_w4 = ref.dequantize_w4a16(packed, scales, zeros, gs)
    try:
        got_w4 = dequantize_w4a16(packed, scales, zeros, gs)
        out["max_abs_err_w4"] = float(np.max(np.abs(want_w4 - got_w4)))
    except Exception as e:
        out.setdefault("_note", f"w4a16 failed: {e}")

    w, ls, gs_scale, gs_nv = ref.generate_nvfp4_fixture()
    want_nv = ref.dequantize_nvfp4(w, ls, gs_scale, gs_nv)
    try:
        got_nv = dequantize_nvfp4(w, ls, gs_scale, gs_nv)
        out["max_abs_err_nv"] = float(np.max(np.abs(want_nv - got_nv)))
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"nvfp4 failed: {e}"

    return out
