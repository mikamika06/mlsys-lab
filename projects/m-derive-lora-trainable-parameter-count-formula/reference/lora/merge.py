import numpy as np


def merge_lora_weights(weight, lora_a, lora_b, alpha, rank):
    scaling = alpha / float(rank)
    return weight + scaling * np.matmul(lora_b, lora_a)
