def predict_tg_throughput(model_bytes, bandwidth_gbps):
    """Predict text generation throughput (tokens/s) based on memory bandwidth roofline."""
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9
    return bandwidth_bytes_per_sec / model_bytes


def compute_throughput_ratio(actual_tg, predicted_tg):
    """Compute ratio of actual tg throughput to predicted roofline tg throughput."""
    if predicted_tg <= 0:
        return 0.0
    return actual_tg / predicted_tg
