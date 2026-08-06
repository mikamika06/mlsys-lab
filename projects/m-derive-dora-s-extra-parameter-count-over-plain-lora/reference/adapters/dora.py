import numpy as np


def compute_extra_parameters(in_features, out_features, rank):
    return rank * (in_features + out_features) + out_features


def dora_forward(x, weight, lora_a, lora_b, magnitude, alpha):
    rank = lora_a.shape[0]
    scaling = alpha / rank
    delta_w = (lora_b @ lora_a) * scaling
    weight_combined = weight + delta_w
    col_norms = np.linalg.norm(weight_combined, axis=1, keepdims=True)
    normalized_weight = weight_combined / (col_norms + 1e-12)
    final_weight = normalized_weight * magnitude
    return x @ final_weight.T
