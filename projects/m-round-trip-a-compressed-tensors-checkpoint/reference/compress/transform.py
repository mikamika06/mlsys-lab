import numpy as np


def round_trip(state_dict):
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, (list, tuple)):
            out[k] = type(v)(round_trip(item) if isinstance(item, dict) else np.array(item) for item in v)
        elif isinstance(v, dict):
            out[k] = round_trip(v)
        else:
            out[k] = v
    return out
