import numpy as np
import ref


def check(workdir):
    from q4_0.quant import quantize, dequantize

    tensors = ref.get_test_tensors()
    errors = []
    out = {"max_abs_err": 1.0}

    try:
        for t in tensors:
            q = quantize(t)
            dq = dequantize(q)
            err = float(np.max(np.abs(t - dq)))
            errors.append(err)

        if errors:
            out["max_abs_err"] = float(max(errors))
    except Exception as e:
        out["_note"] = f"dequantization failed: {type(e).__name__}"

    return out
