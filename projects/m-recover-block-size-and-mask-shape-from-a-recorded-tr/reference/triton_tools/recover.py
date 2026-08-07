import numpy as np


def recover_block_sizes(trace):
    accesses = trace.get("accesses", [])
    if not accesses:
        return {"BLOCK_M": 1, "BLOCK_N": 1}
    m_coords = [a["coord"][0] for a in accesses if "coord" in a and len(a["coord"]) >= 2]
    n_coords = [a["coord"][1] for a in accesses if "coord" in a and len(a["coord"]) >= 2]

    if not m_coords or not n_coords:
        return {"BLOCK_M": int(trace.get("default_block_m", 16)), "BLOCK_N": int(trace.get("default_block_n", 16))}

    unique_m = sorted(list(set(m_coords)))
    unique_n = sorted(list(set(n_coords)))

    block_m = int(np.min(np.diff(unique_m))) if len(unique_m) > 1 else max(1, unique_m[0] + 1)
    block_n = int(np.min(np.diff(unique_n))) if len(unique_n) > 1 else max(1, unique_n[0] + 1)

    if block_m <= 0:
        block_m = 16
    if block_n <= 0:
        block_n = 16

    return {"BLOCK_M": block_m, "BLOCK_N": block_n}


def recover_mask_shape(trace):
    mask = trace.get("mask", None)
    if mask is not None:
        arr = np.array(mask)
        return list(arr.shape)
    accesses = trace.get("accesses", [])
    max_m = 0
    max_n = 0
    for a in accesses:
        c = a.get("coord", [0, 0])
        if len(c) >= 2:
            max_m = max(max_m, c[0])
            max_n = max(max_n, c[1])
    return [max_m + 1, max_n + 1]
