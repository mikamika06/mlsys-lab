import numpy as np

def encode_e4m3(x, scale):
    scaled = x / scale
    clipped = np.clip(scaled, -448.0, 448.0)
    return (np.round(clipped * 8.0) / 8.0).astype(np.float32)

def decode_e4m3(q, scale):
    return q * scale
