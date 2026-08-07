import numpy as np


def compute_percentiles(logs):
    ttfts = []
    itls = []
    for log in logs:
        arrival = log["arrival"]
        tokens = log["tokens"]
        if not tokens:
            continue
        ttft = tokens[0] - arrival
        ttfts.append(ttft)
        for i in range(1, len(tokens)):
            itls.append(tokens[i] - tokens[i - 1])

    res = {}
    if ttfts:
        res["ttft_p50"] = float(np.percentile(ttfts, 50))
        res["ttft_p99"] = float(np.percentile(ttfts, 99))
    else:
        res["ttft_p50"] = 0.0
        res["ttft_p99"] = 0.0

    if itls:
        res["itl_p50"] = float(np.percentile(itls, 50))
        res["itl_p99"] = float(np.percentile(itls, 99))
    else:
        res["itl_p50"] = 0.0
        res["itl_p99"] = 0.0

    return res
