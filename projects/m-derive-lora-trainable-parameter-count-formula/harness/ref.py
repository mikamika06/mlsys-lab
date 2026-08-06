import numpy as np


def param_count(in_features, out_features, rank):
    return int(rank * in_features + out_features * rank)


def merge_weights(weight, lora_a, lora_b, alpha, rank):
    scaling = alpha / float(rank)
    return weight + scaling * np.matmul(lora_b, lora_a)


def lora_forward(x, weight, lora_a, lora_b, alpha, rank):
    scaling = alpha / float(rank)
    base = np.matmul(x, weight.T)
    adapter = np.matmul(np.matmul(x, lora_a.T), lora_b.T)
    return base + scaling * adapter


CONFIGS = [
    {"in_features": 768, "out_features": 768, "rank": 8, "alpha": 16.0},
    {"in_features": 2048, "out_features": 2048, "rank": 16, "alpha": 32.0},
    {"in_features": 4096, "out_features": 11008, "rank": 32, "alpha": 64.0},
]
