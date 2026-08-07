import numpy as np

CASES_DIV = [
    [0.5, 1.5, 2.5, 3.5],
    [0.1, 0.2, 0.6, 0.9],
    [1.0, 2.0, 3.0, 4.0]
]

def ref_count_divergences(arr):
    a = np.round(arr)
    b = np.floor(np.array(arr) + 0.5)
    return int(np.sum(a != b))

CASES_COST = [
    ([{"type": "requantize", "cycles": 50}, {"type": "conv", "cycles": 150}], 200),
    ([{"type": "requantize", "cycles": 0}, {"type": "matmul", "cycles": 500}], 500),
    ([{"type": "requantize", "cycles": 120}, {"type": "requantize", "cycles": 80}], 400)
]

def ref_requant_cost_share(nodes, total_cycles):
    if total_cycles <= 0:
        return 0.0
    rc = sum(n.get("cycles", 0) for n in nodes if n.get("type") == "requantize")
    return float(rc) / float(total_cycles)

CASES_DETECTOR = [
    ([16, 64], 1),
    ([8, 16, 32], -1),
    ([1, 128], 0)
]

def ref_detect_wrong_dimension(shape, dim):
    if dim < 0 or dim >= len(shape):
        return True
    if shape[dim] == 1 and len(shape) > 1:
        return True
    return False
