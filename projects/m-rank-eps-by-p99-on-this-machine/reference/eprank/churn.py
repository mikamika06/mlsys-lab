def should_convert_to_trt(shape_history, threshold):
    unique_shapes = set(tuple(s) for s in shape_history)
    churn_ratio = len(unique_shapes) / max(1, len(shape_history))
    return churn_ratio < threshold
