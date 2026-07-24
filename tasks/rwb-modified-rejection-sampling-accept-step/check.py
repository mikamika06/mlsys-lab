import numpy as np


def _oracle(p, q, draft_token_ids, u_stream):
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


def grade(sol, fx) -> dict:
    p = fx["p"]
    q = fx["q"]
    draft_token_ids = fx["draft_token_ids"]
    u_stream = fx["u_stream"]

    expected = _oracle(p, q, draft_token_ids, u_stream)

    try:
        got = sol.modified_rejection_sample(
            p.copy(), q.copy(), draft_token_ids.copy(), u_stream.copy()
        )
        got = np.asarray(got, dtype=np.int64)
    except Exception:
        return {"exact_match": 0.0}

    if got.shape != expected.shape:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if np.array_equal(got, expected) else 0.0}
