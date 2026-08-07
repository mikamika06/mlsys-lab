def extract_trace_features(event_list):
    if not event_list:
        return {
            "num_calls": 0,
            "total_bytes": 0,
            "avg_bytes_per_call": 0.0,
            "unique_shapes": 0,
        }

    total_bytes = 0
    shapes = set()
    for evt in event_list:
        numel = evt.get("numel", 1)
        elem_size = evt.get("element_size", 4)
        total_bytes += numel * elem_size
        if "shape" in evt:
            shapes.add(tuple(evt["shape"]))

    num_calls = len(event_list)
    return {
        "num_calls": num_calls,
        "total_bytes": total_bytes,
        "avg_bytes_per_call": total_bytes / num_calls if num_calls > 0 else 0.0,
        "unique_shapes": len(shapes),
    }


def classify_communication_pattern(event_list):
    if not event_list:
        return "UNKNOWN"

    for evt in event_list:
        scope = evt.get("scope", "")
        if "tp_group" in scope or "tensor_parallel" in scope:
            return "TENSOR_PARALLEL"
        if "dp_group" in scope or "data_parallel" in scope:
            return "DATA_PARALLEL"

    features = extract_trace_features(event_list)
    if features["num_calls"] >= 8 and features["avg_bytes_per_call"] < 1000000:
        return "TENSOR_PARALLEL"
    if features["unique_shapes"] >= 2 or features["avg_bytes_per_call"] >= 1000000:
        return "DATA_PARALLEL"

    return "DATA_PARALLEL"
