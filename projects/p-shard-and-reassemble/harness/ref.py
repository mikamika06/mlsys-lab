import numpy as np
from gguf_shard.model import Model

def get_test_model_1():
    meta = {"arch": "mock", "vocab": 100}
    tensors = {
        "l1": np.full((10, 10), 0.1, dtype=np.float32),
        "l2": np.full((10, 10), 0.2, dtype=np.float32),
        "l3": np.full((10, 10), 0.3, dtype=np.float32),
    }
    return Model(meta, tensors)

def get_test_model_2():
    meta = {"arch": "mock2"}
    tensors = {
        "a": np.ones((5, 5), dtype=np.int8),
        "b": np.ones((5, 5), dtype=np.float32)
    }
    return Model(meta, tensors)
