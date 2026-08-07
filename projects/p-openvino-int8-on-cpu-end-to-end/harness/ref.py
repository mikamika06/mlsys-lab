import numpy as np

def get_sample_data():
    return [np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(5)]
