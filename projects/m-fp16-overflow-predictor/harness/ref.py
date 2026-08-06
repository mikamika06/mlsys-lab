import numpy as np

CONFIGS = [
    {"name": "layer_block_a", "scale": 12.5, "max_val": 40000.0, "mean": 1.2},
    {"name": "layer_block_b", "scale": 1.2, "max_val": 800.0, "mean": 0.05},
    {"name": "layer_block_c", "scale": 45.0, "max_val": 120000.0, "mean": 3.4},
]

def predict_overflow(tensor_stats, threshold=65504.0):
    peak = tensor_stats.get("max_val", 0.0) * tensor_stats.get("scale", 1.0)
    return bool(peak > threshold)

def rank_sensitivity(tensors):
    scores = []
    for t in tensors:
        score = float(t.get("max_val", 0.0) * abs(t.get("scale", 1.0)))
        scores.append((t["name"], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores]

def generate_golden(tensors):
    return {t["name"]: float(t.get("max_val", 0.0) * t.get("scale", 1.0)) for t in tensors}
