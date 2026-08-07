import numpy as np

TEST_CASES_M1 = [
    {"outlier_ratio": 0.02, "skewness": 1.8, "kl_divergence": 0.6, "min": -10.0, "max": 10.0, "zp_supp": True},
    {"outlier_ratio": 0.005, "skewness": 0.2, "kl_divergence": 0.7, "min": 0.0, "max": 5.0, "zp_supp": True},
    {"outlier_ratio": 0.000, "skewness": 0.1, "kl_divergence": 0.1, "min": -2.0, "max": 2.0, "zp_supp": False},
    {"outlier_ratio": 0.03, "skewness": 0.5, "kl_divergence": 0.2, "min": -1.0, "max": 8.0, "zp_supp": True},
    {"outlier_ratio": 0.001, "skewness": 0.0, "kl_divergence": 0.0, "min": 0.0, "max": 15.0, "zp_supp": True},
]

def pick_calibration_method(tensor_stats):
    has_outliers = tensor_stats.get("outlier_ratio", 0.0) > 0.01
    is_skewed = tensor_stats.get("skewness", 0.0) > 1.5
    kl_divergence_high = tensor_stats.get("kl_divergence", 0.0) > 0.5

    if has_outliers and is_skewed:
        return {"method": "Percentile", "percentile": 99.99}
    elif has_outliers or kl_divergence_high:
        return {"method": "Entropy", "num_bins": 2048}
    else:
        return {"method": "MinMax"}

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

def select_quant_schema(min_val, max_val, has_zero_point_support=True):
    is_strictly_nonnegative = min_val >= 0.0
    symmetric_range = abs(min_val + max_val) < 0.1 * max(abs(min_val), abs(max_val), 1e-6)

    if is_strictly_nonnegative and has_zero_point_support:
        return {"schema": "U8S8", "activation_type": "uint8", "weight_type": "int8", "symmetric": False}
    elif symmetric_range or not has_zero_point_support:
        return {"schema": "S8S8", "activation_type": "int8", "weight_type": "int8", "symmetric": True}
    else:
        return {"schema": "U8S8", "activation_type": "uint8", "weight_type": "int8", "symmetric": False}
