def compute_latency_ratio(ops, tensor_elements):
    threshold = 10.0 + float(tensor_elements) * 0.0001
    if float(ops) > threshold:
        return 2.0
    return 0.8
