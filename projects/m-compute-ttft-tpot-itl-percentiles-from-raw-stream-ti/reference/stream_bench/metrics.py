import numpy as np


def compute_stream_percentiles(streams, percentiles):
    ttfts = []
    tpots = []
    itls = []

    for s in streams:
        arr = s["arrival_time"]
        tok_ts = s["token_timestamps"]
        if not tok_ts:
            continue
        ttfts.append(tok_ts[0] - arr)
        if len(tok_ts) > 1:
            diffs = np.diff(tok_ts)
            itls.extend(diffs.tolist())
            tpots.append((tok_ts[-1] - tok_ts[0]) / (len(tok_ts) - 1))

    res = {}
    p_arr = np.array(percentiles, dtype=float)

    for name, vals in [("ttft", ttfts), ("tpot", tpots), ("itl", itls)]:
        if len(vals) == 0:
            res[name] = {p: 0.0 for p in percentiles}
        else:
            arr_vals = np.array(vals, dtype=float)
            computed = np.percentile(arr_vals, p_arr)
            res[name] = {p: float(computed[i]) for i, p in enumerate(percentiles)}

    return res
