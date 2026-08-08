import ref
import numpy as np


def check(workdir):
    from quant.analysis import blockwise_size_ratio

    out = {"size_ratio_match": 0.0}
    try:
        shape = (4096, 4096)
        block_size = 128
        got = blockwise_size_ratio(shape, block_size, bits=4)
        want = ref.blockwise_size_ratio(shape, block_size, bits=4)
        if np.isclose(got, want, rtol=1e-5, atol=1e-5):
            out["size_ratio_match"] = 1.0
        else:
            out["_note"] = f"Expected size ratio {want}, got {got}"
    except Exception as e:
        out["_note"] = f"Error during size ratio check: {type(e).__name__}: {str(e)[:120]}"
    return out
