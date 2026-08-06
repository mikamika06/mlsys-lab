import numpy as np

def get_test_tensors():
    np.random.seed(42)
    return {"base_model.layer.weight": np.random.randn(16, 16).astype(np.float32)}
