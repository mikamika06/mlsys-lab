def evaluate_fp16_tradeoff(model_def, threshold):
    error = model_def.get("simulated_error", 0.001)
    latency_gain = model_def.get("latency_gain", 0.5)
    feasible = error <= threshold and latency_gain > 0.0
    return {"feasible": feasible, "error": error, "latency_gain": latency_gain}
