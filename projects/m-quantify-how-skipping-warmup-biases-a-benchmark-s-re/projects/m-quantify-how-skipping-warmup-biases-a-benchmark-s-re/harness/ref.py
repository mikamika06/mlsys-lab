import numpy as np

def generate_reference_latencies(prompt_len, seq_len, backend, warmup=5, total=20, seed=42):
    rng = np.random.default_rng(seed + prompt_len + seq_len)
    base = 0.01 + 0.0002 * seq_len
    if backend == "mlx":
        base *= 1.2
    elif backend == "llama_cpp":
        base *= 0.85
    lats = []
    for i in range(total):
        overhead = 0.08 * np.exp(-i / 1.5) if i < warmup else 0.0
        noise = rng.normal(0, 0.001)
        lats.append(max(0.001, base + overhead + noise))
    return lats

def compute_warmup_bias(latencies, warmup=5):
    cold = np.mean(latencies)
    warm = np.mean(latencies[warmup:])
    return float((cold - warm) / warm)

def explain_backend_comparison(mlx_lats, llama_lats, warmup=5):
    m_mean = np.mean(mlx_lats[warmup:])
    l_mean = np.mean(llama_lats[warmup:])
    ratio = m_mean / l_mean
    return {
        "mlx_avg": float(m_mean),
        "llama_avg": float(l_mean),
        "ratio": float(ratio),
        "mlx_slower": ratio > 1.0
    }
