import numpy as np


def get_model():
    rng = np.random.RandomState(42)
    return {
        "fc1": {"type": "fp32", "data": rng.randn(30000000).astype(np.float32) * 0.1},
        "fc2": {"type": "fp32", "data": rng.randn(10000000).astype(np.float32) * 0.1},
        "fc3": {"type": "fp32", "data": rng.randn(5000000).astype(np.float32) * 0.1},
    }


def decompress(encoded):
    if encoded["type"] == "fp32":
        return encoded["data"]
    elif encoded["type"] == "palette":
        return encoded["palette"][encoded["indices"]]
    elif encoded["type"] == "uint8":
        return (encoded["data"].astype(np.float32) - encoded["zp"]) * encoded["scale"]
    elif encoded["type"] == "sparse":
        arr = np.zeros(np.prod(encoded["shape"]), dtype=np.float32)
        arr[encoded["indices"]] = encoded["data"]
        return arr.reshape(encoded["shape"])


def get_sizes(state_dict):
    total = 0
    for k, v in state_dict.items():
        if v["type"] == "fp32":
            total += v["data"].size * 4
        elif v["type"] == "palette":
            total += v["indices"].size * 1 + v["palette"].size * 4
        elif v["type"] == "uint8":
            total += v["data"].size * 1 + 8
        elif v["type"] == "sparse":
            total += v["indices"].size * 4 + v["data"].size * 4
    return total


def evaluate(orig, comp):
    acc = 85.0
    sens = {"fc1": 1.0, "fc2": 2.0, "fc3": 100.0}
    for k in orig:
        c = comp[k]
        c_dec = decompress(c)
        mse = float(np.mean((orig[k]["data"] - c_dec)**2))
        acc -= mse * sens[k]
    return acc
