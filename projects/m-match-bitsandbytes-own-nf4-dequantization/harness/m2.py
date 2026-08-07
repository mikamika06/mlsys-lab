import ref
import numpy as np


def check(workdir):
    from nf4.dequant import dequantize_nf4

    out = {"max_abs_err": 999.0}

    try:
        rng = np.random.default_rng(42)
        packed = rng.integers(0, 256, size=256, dtype=np.uint8)
        absmax = rng.uniform(0.1, 2.5, size=8).astype(np.float32)

        got = dequantize_nf4(packed, absmax, blocksize=64)
        want = ref.dequantize_nf4(packed, absmax, blocksize=64)

        if got.shape == want.shape:
            out["max_abs_err"] = float(np.max(np.abs(got - want)))
        else:
            out["_note"] = f"shape mismatch: got {got.shape}, want {want.shape}"
    except Exception as e:
        out["_note"] = f"dequantize error: {type(e).__name__}: {str(e)}"

    return out
