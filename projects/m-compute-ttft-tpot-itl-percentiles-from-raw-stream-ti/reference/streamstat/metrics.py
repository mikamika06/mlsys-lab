import numpy as np

def compute_percentiles(raw_streams):
    ttfts = []
    tpots = []
    itls = []
    for stream in raw_streams:
        _, _, t_sent, timestamps = stream
        if not timestamps:
            continue
        ttfts.append(timestamps[0] - t_sent)
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            itls.extend(list(intervals))
            tpots.append((timestamps[-1] - timestamps[0]) / (len(timestamps) - 1))
        else:
            tpots.append(0.0)
    return {
        "ttft_p50": float(np.percentile(ttfts, 50)) if ttfts else 0.0,
        "ttft_p99": float(np.percentile(ttfts, 99)) if ttfts else 0.0,
        "tpot_p50": float(np.percentile(tpots, 50)) if tpots else 0.0,
        "tpot_p99": float(np.percentile(tpots, 99)) if tpots else 0.0,
        "itl_p50": float(np.percentile(itls, 50)) if itls else 0.0,
        "itl_p99": float(np.percentile(itls, 99)) if itls else 0.0,
    }
