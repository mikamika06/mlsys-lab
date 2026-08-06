def predict_tg_throughput(model_bytes, bandwidth_gbps):
    """Predict text generation throughput (tokens/s) based on memory bandwidth roofline."""
    raise NotImplementedError


def compute_throughput_ratio(actual_tg, predicted_tg):
    """Compute ratio of actual tg throughput to predicted roofline tg throughput."""
    raise NotImplementedError
