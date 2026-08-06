import numpy as np

def evaluate_fp16(graph, threshold=1e-3):
    weights = graph["weights"]
    fp16_weights = weights.astype(np.float16).astype(np.float32)
    error = float(np.max(np.abs(weights - fp16_weights)))
    latency_ratio = 0.5
    return {"error": error, "latency_ratio": latency_ratio, "valid": error <= threshold}
