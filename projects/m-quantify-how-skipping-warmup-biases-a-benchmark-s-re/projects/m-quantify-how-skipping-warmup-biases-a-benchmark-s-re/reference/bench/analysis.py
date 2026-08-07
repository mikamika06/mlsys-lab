import numpy as np

def quantify_bias(latencies, warmup=5):
    if len(latencies) <= warmup:
        return 0.0
    cold = np.mean(latencies)
    warm = np.mean(latencies[warmup:])
    return float((cold - warm) / warm)

def explain_mlx_vs_llama(mlx_latencies, llama_latencies, warmup=5):
    m_mean = np.mean(mlx_latencies[warmup:])
    l_mean = np.mean(llama_latencies[warmup:])
    ratio = m_mean / l_mean
    return {
        "mlx_avg": float(m_mean),
        "llama_avg": float(l_mean),
        "ratio": float(ratio),
        "mlx_slower": ratio > 1.0
    }
