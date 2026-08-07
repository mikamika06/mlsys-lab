import numpy as np


def fake_quantize(W):
    return np.round(W * 4.0) / 4.0


def loftq_init(W, rank):
    W_q = fake_quantize(W)
    residual = W - W_q
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    sqrt_S = np.sqrt(S_r)
    A = np.diag(sqrt_S) @ Vt_r
    B = U_r @ np.diag(sqrt_S)
    return W_q, A, B


def dora_forward(x, W_q, A, B, g, use_dora):
    combined = W_q + B @ A
    if use_dora:
        col_norms = np.linalg.norm(combined, axis=1, keepdims=True)
        col_norms = np.maximum(col_norms, 1e-8)
        normalized_weight = combined / col_norms
        weight = g[:, None] * normalized_weight
        return x @ weight.T
    else:
        return x @ W_q.T + x @ A.T @ B.T


def trainable_param_delta(in_features, out_features, rank, use_dora):
    lora_params = rank * (in_features + out_features)
    dora_params = lora_params + out_features if use_dora else lora_params
    return dora_params - lora_params


CASES = [
    {"W": np.random.RandomState(42).randn(16, 32), "rank": 4, "x": np.random.RandomState(43).randn(5, 32), "g": np.ones(16)},
    {"W": np.random.RandomState(44).randn(24, 20), "rank": 8, "x": np.random.RandomState(45).randn(3, 20), "g": np.ones(24) * 1.1},
]
