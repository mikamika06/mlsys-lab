import numpy as np


def attention_sink_mass(logits: np.ndarray, k: int) -> float:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    attn = np.exp(x)
    attn = attn / np.sum(attn, axis=1, keepdims=True)

    column_mass = np.sum(attn, axis=0)
    return float(np.sum(column_mass[:k]) / np.sum(column_mass))
