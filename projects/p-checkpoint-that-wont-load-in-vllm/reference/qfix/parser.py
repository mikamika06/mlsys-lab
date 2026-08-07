import numpy as np

def parse_checkpoint(raw_data):
    if not isinstance(raw_data, dict):
        raise ValueError("Invalid format")
    return {k: np.array(v) for k, v in raw_data.items()}
