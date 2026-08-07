import ref
import numpy as np


def check(workdir):
    from ternary.tq1 import pack_tq1_0, unpack_tq1_0
    vals = ref.get_test_values()
    out = {"pack_unpack_exact": 0.0, "max_rel_err": 1.0}
    try:
        packed = pack_tq1_0(vals)
        unpacked = unpack_tq1_0(packed, len(vals))
        exact = np.array_equal(vals, unpacked)
        out["pack_unpack_exact"] = 1.0 if exact else 0.0
        diff = np.abs(vals.astype(np.float32) - unpacked.astype(np.float32))
        max_err = float(np.max(diff))
        out["max_rel_err"] = max_err
    except Exception as e:
        out["_note"] = f"m1 execution error: {type(e).__name__}: {str(e)[:100]}"
    return out
