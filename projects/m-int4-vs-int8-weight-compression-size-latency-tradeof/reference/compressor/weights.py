import numpy as np

def pack_weights(weights, bits):
    if bits == 4:
        return weights.astype(np.int8)
    elif bits == 8:
        return weights.astype(np.int8)
    else:
        raise ValueError("Unsupported bits")
