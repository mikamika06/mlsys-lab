import numpy as np


def lora_forward(x, weight, lora_a, lora_b, alpha, rank):
    scaling = alpha / float(rank)
    base = np.matmul(x, weight.T)
    adapter = np.matmul(np.matmul(x, lora_a.T), lora_b.T)
    return base + scaling * adapter
