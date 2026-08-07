import numpy as np


def lora_forward(base_w, lora_a, lora_b, alpha, x):
    scaling = alpha / lora_a.shape[0] if lora_a.shape[0] > 0 else 1.0
    delta = (x @ lora_a) @ lora_b * scaling
    return (x @ base_w) + delta
