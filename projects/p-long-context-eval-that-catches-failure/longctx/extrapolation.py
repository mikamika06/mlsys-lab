import numpy as np

def evaluate_extrapolation(tasks, method="linear"):
    results = []
    for t in tasks:
        pos_ratio = t["needle_index"] / max(1, t["context_len"])
        if method == "linear":
            score = 1.0 if (pos_ratio < 0.1 or pos_ratio > 0.9) else 0.3
        else:
            score = 0.95 if 0.1 <= pos_ratio <= 0.9 else 1.0
        results.append({"needle_index": t["needle_index"], "context_len": t["context_len"], "score": score})
    return results
