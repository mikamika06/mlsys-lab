import numpy as np

def detect_scale_collapse(histogram, minmax_range, target_range):
    counts, bin_edges = histogram
    total_samples = np.sum(counts)
    if total_samples == 0:
        return {"collapsed": False, "effective_bins": 0, "scale_ratio": 1.0}

    min_val, max_val = target_range
    in_range_mask = (bin_edges[:-1] >= min_val) & (bin_edges[1:] <= max_val)
    active_bins = np.sum(counts[in_range_mask] > 0)

    full_span = minmax_range[1] - minmax_range[0]
    target_span = max_val - min_val
    scale_ratio = target_span / full_span if full_span > 0 else 1.0

    collapsed = bool(active_bins < 4 or scale_ratio < 0.1)
    return {
        "collapsed": collapsed,
        "effective_bins": int(active_bins),
        "scale_ratio": float(scale_ratio)
    }
