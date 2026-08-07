import numpy as np
import ref


def check(workdir):
    try:
        from quant.formats import nvfp4, effective_bits
    except ImportError:
        return {"_note": "ImportError on quant.formats"}

    out = {"nvfp4_match": 0.0, "bits_match": 0.0}

    np.random.seed(42)
    x = np.random.randn(1024) * 10
    x[0:16] = 0.01
    x[16:32] = 100.0

    if np.allclose(ref.nvfp4(x), nvfp4(x)):
        out["nvfp4_match"] = 1.0

    try:
        mx_bits = effective_bits("mxfp4")
        nv_bits = effective_bits("nvfp4")
        if np.isclose(mx_bits, 4.25) and np.isclose(nv_bits, 4.28125):
            out["bits_match"] = 1.0
    except Exception:
        pass

    return out
