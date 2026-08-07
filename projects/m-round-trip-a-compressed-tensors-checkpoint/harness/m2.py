import numpy as np
import ref


def check(workdir):
    from compress.transform import round_trip

    out = {"max_abs_err": 999.0}
    try:
        processed = round_trip(ref.SYNTHETIC_CHECKPOINT)
        max_err = 0.0
        for k, v in ref.SYNTHETIC_CHECKPOINT.items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
                err = np.max(np.abs(processed[k] - v))
                if err > max_err:
                    max_err = err
            elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.integer):
                diff = np.max(np.abs(processed[k].astype(np.float32) - v.astype(np.float32)))
                if diff > max_err:
                    max_err = diff
        out["max_abs_err"] = float(max_err)
    except Exception as e:
        out["_note"] = f"Round trip failed: {e}"
    return out
