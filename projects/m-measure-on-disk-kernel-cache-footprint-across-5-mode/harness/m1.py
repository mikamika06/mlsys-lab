import os
import tempfile
import ref


def check(workdir):
    from cachefootprint.measure import measure_footprint

    out = {"sizes_matched": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            res = measure_footprint(ref.MODEL_SIZES, tmp)
            matched = 0
            for size in ref.MODEL_SIZES:
                if isinstance(res, dict) and size in res and res[size] > 0:
                    matched += 1
            out["sizes_matched"] = float(matched)
        except Exception as e:
            out["_note"] = f"measure_footprint failed: {type(e).__name__}: {str(e)[:120]}"
    return out
