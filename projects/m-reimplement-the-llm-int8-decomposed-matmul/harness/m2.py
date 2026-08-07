import ref
import numpy as np


def check(workdir):
    from int8_matmul.dequant import derive_vector_scales
    np.random.seed(99)
    tensor = np.random.randn(10, 20)
    want = ref.ref_derive_scales(tensor)
    try:
        got = derive_vector_scales(tensor)
    except Exception as e:
        return {"scales_matched": 0.0, "_note": f"raised {type(e).__name__}: {e}"}

    got_arr = np.array(got)
    if got_arr.shape != want.shape:
        return {"scales_matched": 0.0, "_note": f"shape mismatch: got {got_arr.shape}, want {want.shape}"}

    match = 1.0 if np.allclose(got_arr, want, atol=1e-5) else 0.0
    out = {"scales_matched": match}
    if match == 0.0:
        out["_note"] = f"scales differ"
    return out
