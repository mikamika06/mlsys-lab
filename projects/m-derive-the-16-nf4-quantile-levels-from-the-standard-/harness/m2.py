import numpy as np
import ref


def check(workdir):
    from nf4.block import quantize_nf4_blockwise
    out = {"max_abs_err": 1.0}
    try:
        np.random.seed(42)
        x = np.random.randn(1024).astype(np.float64)
        got = quantize_nf4_blockwise(x, block_size=64)
        want = ref.quantize_nf4_blockwise(x, block_size=64)
        err = np.max(np.abs(got - want))
        out["max_abs_err"] = float(err)
        if err > 0.15:
            out["_note"] = f"max absolute error too high: {err}"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)}"
    return out
