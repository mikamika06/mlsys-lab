import ref
import numpy as np


def check(workdir):
    from decode_prof.analysis import diagnose_occupancy_limiter
    out = {"occupancy_matched": 0.0}
    try:
        got = diagnose_occupancy_limiter(ref.BATCH_SIZES, ref.BW_REAL, ref.PEAK_BW)
        if got is not None and len(got) == len(ref.OCCUPANCY_METRICS):
            if np.allclose(got, ref.OCCUPANCY_METRICS, atol=1e-5):
                out["occupancy_matched"] = 1.0
            else:
                out["_note"] = f"occupancy metrics mismatch: got {got}, expected {ref.OCCUPANCY_METRICS}"
        else:
            out["_note"] = "invalid return from diagnose_occupancy_limiter"
    except Exception as e:
        out["_note"] = f"error in diagnose_occupancy_limiter: {type(e).__name__}: {str(e)[:120]}"
    return out
