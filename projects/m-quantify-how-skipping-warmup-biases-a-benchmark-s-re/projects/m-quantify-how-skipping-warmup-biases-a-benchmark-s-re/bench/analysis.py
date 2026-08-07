import numpy as np

def quantify_bias(latencies, warmup=5):
    if len(latencies) <= warmup:
        return 0.0
    cold_avg = np.mean(latencies)
    warm_avg = np.mean(latencies[warmup:])
    return float((cold_avg - warm_avg) / warm_avg)

def explain_mlx_vs_llama(mlx_latencies, llama_latencies, warmup=5):
    mlx_warm = np.mean(mlx_latencies[warmup:])
    llama_warm = np.mean(llama_latencies[warmup:])
    ratio = mlx_warm / llama_warm
    return {"mlx_avg": float(mlx_warm), "llama_avg": float(llama_warm), "ratio": float(ratio), "mlx_slower": ratio > 1.0}
