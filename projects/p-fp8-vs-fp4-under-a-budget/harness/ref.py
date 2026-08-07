import numpy as np

def get_sample_weights():
    np.random.seed(42)
    return np.random.randn(128).astype(np.float32)

def get_validation_data():
    return np.ones(32, dtype=np.float32)
