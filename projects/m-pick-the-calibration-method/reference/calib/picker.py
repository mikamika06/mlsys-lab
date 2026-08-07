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
