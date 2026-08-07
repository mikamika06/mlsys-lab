import numpy as np

def get_sample_model():
    return {
        "blocks": [
            {"device": "GPU", "fallback_op": "custom_slice"},
            {"device": "GPU", "fallback_op": "dynamic_shape"}
        ],
        "optimized": False
    }

def get_sample_input():
    return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
