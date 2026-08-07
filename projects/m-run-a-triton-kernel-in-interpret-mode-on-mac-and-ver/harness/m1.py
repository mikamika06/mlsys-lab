import ref
import numpy as np


def check(workdir):
    from triton_profiler.interpret import run_interpreted_kernel
    x, y, bs = ref.get_test_inputs()

    try:
        got = run_interpreted_kernel(x, y, bs)
    except Exception as e:
        return {"blocks_matched": 0.0, "max_abs_err": 999.0, "_note": f"raised {type(e).__name__}: {str(e)[:100]}"}

    from triton_profiler.interpret import run_interpreted_kernel as ref_fn
    # compute with internal oracle logic directly or via reference module
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    out_ref = x_arr + y_arr

    got_out = np.asarray(got.get("output", []), dtype=np.float32)

    if got_out.shape != out_ref.shape:
        return {"blocks_matched": 0.0, "max_abs_err": 999.0, "_note": "shape mismatch"}

    max_err = float(np.max(np.abs(got_out - out_ref)))

    n = len(x)
    expected_blocks = (n + bs - 1) // bs
    got_blocks = len(got.get("per_block_times", []))

    blocks_matched = 1.0 if got_blocks == expected_blocks else 0.0
    return {"blocks_matched": blocks_matched, "max_abs_err": max_err}
