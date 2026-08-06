import ref
import numpy as np

def check(workdir):
    from e2m1.quantize import quantize_e2m1
    out = {"quantization_matched": 0.0}
    try:
        x = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, -1.0], dtype=np.float32)
        got = quantize_e2m1(x)
        want = ref.quantize_e2m1(x)
        if np.allclose(got, want, atol=1e-5):
            out["quantization_matched"] = 1.0
        else:
            out["_note"] = "quantization mismatch"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}"
    return out
