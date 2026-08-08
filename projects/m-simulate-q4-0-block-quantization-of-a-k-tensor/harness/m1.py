import numpy as np
import ref


def check(workdir):
    from q4_0.quant import quantize

    tensors = ref.get_test_tensors()
    ref_quants = ref.get_reference_quantizations()

    out = {"shapes_matched": 1.0, "scales_matched": 1.0, "bytes_matched": 1.0}

    for i, (t, r_q) in enumerate(zip(tensors, ref_quants)):
        try:
            got = quantize(t)
            if not isinstance(got, dict):
                raise ValueError("returned output is not a dictionary")

            if got.get("shape") != r_q["shape"]:
                out["shapes_matched"] = 0.0
            if not np.allclose(got.get("scales", []), r_q["scales"], atol=1e-5):
                out["scales_matched"] = 0.0
            if not np.array_equal(got.get("packed", []), r_q["packed"]):
                out["bytes_matched"] = 0.0
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"tensor {i} failed: {type(e).__name__}"
            out["shapes_matched"] = 0.0
            out["scales_matched"] = 0.0
            out["bytes_matched"] = 0.0

    return out
