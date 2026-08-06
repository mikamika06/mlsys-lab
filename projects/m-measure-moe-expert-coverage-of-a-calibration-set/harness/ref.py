CONFIGS = [
    {"routing": [[0, 1], [1, 2], [0, 2], [1, 3]], "num_experts": 4},
    {"routing": [[0], [0], [0], [0]], "num_experts": 4},
    {"routing": [[0, 1, 2, 3], [0, 1, 2, 3]], "num_experts": 4},
]

COMPARISONS = [
    ({"layer.0": [1.0, 2.0, 3.0]}, {"layer.0": [1.0, 2.0, 3.0]}),
    ({"layer.0": [1.0, 0.0]}, {"layer.0": [0.0, 1.0]}),
    ({"layer.0": [2.0, 4.0]}, {"layer.0": [1.0, 2.0]}),
]

TRUNCATIONS = [
    ({"layers": [{"data": [1, 2]}, {"data": [3, 4]}]}, 2, False),
    ({"layers": [{"data": [1, 2]}]}, 2, True),
    ({"layers": [{"data": []}, {"data": [3, 4]}]}, 2, True),
]

def measure_coverage(routing_data, num_experts):
    counts = {i: 0 for i in range(num_experts)}
    total_tokens = len(routing_data)
    for experts in routing_data:
        for e in set(experts):
            if 0 <= e < num_experts:
                counts[e] += 1
    coverage_ratio = sum(1 for e, c in counts.items() if c > 0) / num_experts
    return {"counts": counts, "total_tokens": total_tokens, "coverage_ratio": coverage_ratio}

def compare_imatrices(imatrix_a, imatrix_b):
    import numpy as np
    keys = sorted(set(imatrix_a.keys()) & set(imatrix_b.keys()))
    if not keys:
        return 0.0
    diffs = []
    for k in keys:
        a = np.array(imatrix_a[k], dtype=float)
        b = np.array(imatrix_b[k], dtype=float)
        if a.shape != b.shape:
            continue
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            diffs.append(0.0 if np.array_equal(a, b) else 1.0)
        else:
            cos_sim = np.dot(a, b) / (norm_a * norm_b)
            diffs.append(float(cos_sim))
    return float(np.mean(diffs)) if diffs else 0.0

def detect_truncation(imatrix_data, expected_layers):
    if not isinstance(imatrix_data, dict):
        return True
    layers = imatrix_data.get("layers", [])
    if len(layers) < expected_layers:
        return True
    for l in layers:
        if "data" not in l or not l["data"]:
            return True
    return False
