import numpy as np


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
    sizes = {}
    for k, v in state_dict.items():
        if v["type"] == "fp32":
            sizes[k] = v["data"].size * 4
        elif v["type"] == "palette":
            sizes[k] = v["indices"].size * 1 + v["palette"].size * 4
        elif v["type"] == "uint8":
            sizes[k] = v["data"].size * 1 + 8
        elif v["type"] == "sparse":
            sizes[k] = v["indices"].size * 4 + v["data"].size * 4
    return sizes, sum(sizes.values())


def palettize(tensor, k=256):
    t_min, t_max = float(tensor.min()), float(tensor.max())
    if t_min == t_max:
        return {"type": "palette", "indices": np.zeros_like(tensor, dtype=np.uint8), "palette": np.array([t_min], dtype=np.float32)}
    palette = np.linspace(t_min, t_max, k, dtype=np.float32)
    norm = (tensor - t_min) / (t_max - t_min)
    indices = np.clip(np.round(norm * (k - 1)), 0, k - 1).astype(np.uint8)
    return {"type": "palette", "indices": indices, "palette": palette}


def quantize(tensor):
    t_min, t_max = float(tensor.min()), float(tensor.max())
    scale = (t_max - t_min) / 255.0
    if scale == 0:
        scale = 1.0
    zp = int(np.clip(np.round(-t_min / scale), 0, 255))
    q = np.clip(np.round(tensor / scale + zp), 0, 255).astype(np.uint8)
    return {"type": "uint8", "data": q, "scale": float(scale), "zp": zp}


def sparsify(tensor, threshold=0.05):
    mask = np.abs(tensor) > threshold
    return {
        "type": "sparse",
        "shape": tensor.shape,
        "indices": np.where(mask.flatten())[0].astype(np.int32),
        "data": tensor[mask].astype(np.float32)
    }


def compress_model(state_dict):
    return {
        "fc1": quantize(state_dict["fc1"]["data"]),
        "fc2": sparsify(state_dict["fc2"]["data"], threshold=0.25),
        "fc3": sparsify(state_dict["fc3"]["data"], threshold=0.15)
    }


def pareto_frontier(points):
    optimal = []
    for s, a in points:
        dominated = False
        for s_other, a_other in points:
            if (s_other <= s and a_other >= a) and (s_other < s or a_other > a):
                dominated = True
                break
        if not dominated:
            optimal.append((s, a))
    return sorted(list(set(optimal)), key=lambda x: x[0])
