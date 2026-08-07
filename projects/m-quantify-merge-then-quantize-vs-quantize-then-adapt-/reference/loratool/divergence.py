import numpy as np

def measure_divergence(config):
    np.random.seed(42)
    dim = config["hidden_dim"]
    W = np.random.randn(dim, dim).astype(np.float32) * 0.1
    A = np.random.randn(config["rank"], dim).astype(np.float32) * 0.05
    B = np.random.randn(dim, config["rank"]).astype(np.float32) * 0.05
    delta = (B @ A) * (config["alpha"] / config["rank"])

    W_merged = W + delta
    scale_factor = np.max(np.abs(W_merged)) / 7.0
    W_merged_q = np.round(W_merged / scale_factor).clip(-8, 7) * scale_factor

    scale_base = np.max(np.abs(W)) / 7.0
    W_q = np.round(W / scale_base).clip(-8, 7) * scale_base
    W_q_adapted = W_q + delta

    diff = np.linalg.norm(W_merged_q - W_q_adapted) / (np.linalg.norm(W_q_adapted) + 1e-8)
    return float(diff)
