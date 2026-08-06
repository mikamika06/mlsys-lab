import ref
import numpy as np

def check(workdir):
    from epall.log import reconstruct_token_counts
    out = {"reconstruction_match": 0.0}
    ok = 0
    for logs, ws in ref.LOG_SAMPLES:
        want = ref.reconstruct_counts(logs, ws)
        got = reconstruct_token_counts(logs, ws)
        if np.array_equal(want, got):
            ok += 1
    out["reconstruction_match"] = 1.0 if ok == len(ref.LOG_SAMPLES) else 0.0
    return out
