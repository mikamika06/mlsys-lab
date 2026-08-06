import numpy as np
import ref


def check(workdir):
    from mxfp4.decode import decode_mxfp4_block

    test_blocks = ref.generate_test_blocks()
    out = {"blocks_matched": 0.0, "max_abs_err": 1e9}

    matched = 0
    max_err = 0.0

    for scale, nibbles in test_blocks:
        want = ref.ref_decode_mxfp4_block(scale, nibbles)
        try:
            got = decode_mxfp4_block(scale, nibbles)
            err = float(np.max(np.abs(want - got)))
            if err > max_err:
                max_err = err
            if np.allclose(want, got, atol=1e-6):
                matched += 1
        except Exception as e:
            out["_note"] = f"decode_mxfp4_block failed: {e}"
            return out

    out["blocks_matched"] = float(matched)
    out["max_abs_err"] = float(max_err)
    return out
