import numpy as np


def modified_rejection_sample(p, q, draft_token_ids, u_stream):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    draft_token_ids = np.asarray(draft_token_ids, dtype=np.int64)
    u_stream = np.asarray(u_stream, dtype=np.float64)

    T, V = p.shape
    ptr = 0
    out = np.empty(T, dtype=np.int64)

    for t in range(T):
        u_accept = u_stream[ptr]
        ptr += 1

        tok = int(draft_token_ids[t])
        denom = q[t, tok]
        ratio = min(1.0, p[t, tok] / denom) if denom > 0 else 0.0

        if u_accept <= ratio:
            out[t] = tok
        else:
            r = np.maximum(p[t] - q[t], 0.0)
            r = r / r.sum()
            u_resample = u_stream[ptr]
            ptr += 1
            cdf = np.cumsum(r)
            idx = int(np.searchsorted(cdf, u_resample, side="left"))
            idx = min(idx, V - 1)
            out[t] = idx

    return out
