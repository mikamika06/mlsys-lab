import numpy as np

def get_test_tensors():
    rng = np.random.default_rng(42)
    return [
        rng.standard_normal((16, 16), dtype=np.float32) * 10.0,
        rng.standard_normal((32, 64), dtype=np.float32) * 0.5,
        np.array([[-100.0, 0.0, 200.0], [50.0, -20.0, 10.0]], dtype=np.float32)
    ]

def encode_e4m3(x, scale):
    scaled = x / scale
    clipped = np.clip(scaled, -448.0, 448.0)
    return (np.round(clipped * 8.0) / 8.0).astype(np.float32)

def decode_e4m3(q, scale):
    return q * scale

def descale_tensor(q, scale):
    return q * scale

def optimize_scale(x):
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return 1.0
    return float(max_val / 448.0)
