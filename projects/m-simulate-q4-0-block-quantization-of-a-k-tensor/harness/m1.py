import ref
import numpy as np


def check(workdir):
    from quant.q4_0 import quantize_q4_0, dequantize_q4_0

    out = {"blocks_matched": 0.0, "tensors": float(len(ref.TENSORS))}
    ok = 0
    for i, t in enumerate(ref.TENSORS):
        try:
            got_bytes = quantize_q4_0(t)
            want_bytes = ref.quantize_q4_0(t)
            if got_bytes != want_bytes:
                if "_note" not in out:
                    out["_note"] = f"tensor {i} byte mismatch"
                continue
            got_arr = dequantize_q4_0(got_bytes, t.shape)
            want_arr = ref.dequantize_q4_0(want_bytes, t.shape)
            if np.allclose(got_arr, want_arr, atol=1e-5):
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"tensor {i} array mismatch after dequantize"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"tensor {i} raised {type(e).__name__}: {e}"
    out["blocks_matched"] = float(ok)
    return out
