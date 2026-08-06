import numpy as np


def reconstruct_scales(nodes):
    scales = []
    for node in nodes:
        mn = float(node["min_val"])
        mx = float(node["max_val"])
        levels = int(node.get("levels", 256))
        scale = (mx - mn) / (levels - 1) if levels > 1 else 1.0
        if scale == 0.0:
            scale = 1.0
        scales.append(scale)
    return scales


def classify_collapse(sizes, variances, threshold):
    results = []
    for s, v in zip(sizes, variances):
        score = float(v) / max(float(s), 1.0)
        results.append("collapse" if score > float(threshold) else "stable")
    return results
