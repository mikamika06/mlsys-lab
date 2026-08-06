import numpy as np


def get_rslora_scaling(alpha, rank):
    return alpha / np.sqrt(rank)
