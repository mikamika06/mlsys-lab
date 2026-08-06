import ref
import numpy as np

def check(workdir):
    from goodput.logs import reconstruct_batch_sizes
    out = {"batch_match": 0.0}
    try:
        want = ref.reconstruct_batch_sizes(ref.EVENTS, ref.TIME_GRID)
        got = reconstruct_batch_sizes(ref.EVENTS, ref.TIME_GRID)
        if np.allclose(want, got):
            out["batch_match"] = 1.0
        else:
            out["_note"] = f"got batch sizes {got}, want {want}"
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)[:100]}"
    return out
