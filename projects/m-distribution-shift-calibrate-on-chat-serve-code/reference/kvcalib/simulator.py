import numpy as np
from kvcalib.metrics import compute_rel_err


def simulate_shift(scales, activations):
    clipped = np.clip(activations, -scales * 7.0, scales * 7.0)
    quantized = np.round(clipped / scales) * scales
    return compute_rel_err(quantized, activations)
