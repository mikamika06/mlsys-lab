import ref
import numpy as np


def check(workdir):
    out = {"packing_matched": 0.0}
    try:
        from woq.packing import pack_int4_groups, unpack_int4_groups
    except Exception as e:
        out["_note"] = f"import failed: {e}"
        return out

    weights = ref.generate_test_data()
    group_size = 32
    try:
        packed, scales = pack_int4_groups(weights, group_size)
        recovered = unpack_int4_groups(packed, scales, group_size, weights.shape)
    except Exception as e:
        out["_note"] = f"execution failed: {e}"
        return out

    if recovered.shape == weights.shape and np.all(np.isfinite(recovered)):
        diff = np.abs(weights - recovered)
        if np.mean(diff) < 1.0:
            out["packing_matched"] = 1.0
        else:
            out["_note"] = f"mean reconstruction error too high: {np.mean(diff)}"
    else:
        out["_note"] = "shape mismatch or non-finite values in recovery"
    return out
