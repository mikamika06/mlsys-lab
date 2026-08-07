import numpy as np

def get_sample_data():
    np.random.seed(42)
    raw = {"layers.0.weight": np.array([0x32, 0x54], dtype=np.uint8)}
    inputs = np.array([1.0, 2.0], dtype=np.float32)
    return raw, inputs
