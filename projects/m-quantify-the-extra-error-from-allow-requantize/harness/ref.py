import numpy as np


def get_sample_weights():
    np.random.seed(42)
    return list(np.random.randn(32).astype(float))


def get_sample_model():
    return {
        "header": "GGUF_VALID",
        "tensors": {
            "layer_0.weight": [0.1, 0.2],
            "layer_2.weight": [0.5, 0.6]
        }
    }
