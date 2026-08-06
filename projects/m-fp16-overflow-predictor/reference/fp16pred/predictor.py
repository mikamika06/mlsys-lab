def predict_overflow(tensor_stats, threshold=65504.0):
    peak = tensor_stats.get("max_val", 0.0) * tensor_stats.get("scale", 1.0)
    return bool(peak > threshold)
