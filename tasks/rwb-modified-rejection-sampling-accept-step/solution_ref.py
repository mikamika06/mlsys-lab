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
            r_sum = 0.0
            r_vals = []
            for i in range(V):
                diff = p[t, i] - q[t, i]
                val = diff if diff > 0.0 else 0.0
                r_sum += val
                r_vals.append(val)

            r_normalized = [val / r_sum for val in r_vals]

            u_resample = u_stream[ptr]
            ptr += 1

            cdf_val = 0.0
            idx = V
            for i in range(V):
                cdf_val += r_normalized[i]
                if cdf_val >= u_resample:
                    idx = i
                    break

            idx = min(idx, V - 1)
            out[t] = idx

    return out
