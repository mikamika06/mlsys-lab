import numpy as np


def generate_inputs(B=2, H=4, N=128, D=32, seed=1337):
    np.random.seed(seed)
    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)
    sm_scale = 1.0 / np.sqrt(D)
    return Q, K, V, sm_scale


def standard_attention_forward(Q, K, V, sm_scale, causal=False):
    B, H, N, D = Q.shape
    Q64 = Q.astype(np.float64)
    K64 = K.astype(np.float64)
    V64 = V.astype(np.float64)

    S = np.matmul(Q64, K64.transpose(0, 1, 3, 2)) * sm_scale
    if causal:
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        S[:, :, mask] = -np.inf

    S_max = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    P = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    O = np.matmul(P, V64)
    return O


def compute_attention_flops(B, H, N, D, causal=False):
    multiplier = 2 if causal else 4
    return float(multiplier * B * H * N * N * D)


def derive_tflops(B, H, N, D, wall_clock_seconds, causal=False):
    total_flops = compute_attention_flops(B, H, N, D, causal=causal)
    return float((total_flops / wall_clock_seconds) / 1e12)
