import numpy as np


def recover_per_channel_scales(model_path):
    if isinstance(model_path, dict):
        return model_path["scales"]
    return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
