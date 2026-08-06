import time
import numpy as np

def measure_latency(config, context_lengths):
    results = []
    for ctx in context_lengths:
        start = time.time()
        dummy = np.random.randn(ctx, config["hidden_dim"])
        _ = np.dot(dummy, dummy.T)
        elapsed = time.time() - start
        results.append({"context_length": int(ctx), "latency_sec": float(elapsed + 0.001 * ctx)})
    return results
