import numpy as np

def evaluate_curve(tasks, model_type="flawed"):
    results = []
    for t in tasks:
        pos = t["position"]
        if model_type == "flawed":
            if 0.2 <= pos <= 0.8:
                acc = float(np.random.uniform(0.0, 0.2))
            else:
                acc = float(np.random.uniform(0.9, 1.0))
        else:
            acc = float(np.random.uniform(0.9, 1.0))
        results.append({"position": pos, "accuracy": acc})
    return results

def detect_dip(curve_results):
    mid_accs = [r["accuracy"] for r in curve_results if 0.2 <= r["position"] <= 0.8]
    edge_accs = [r["accuracy"] for r in curve_results if r["position"] < 0.2 or r["position"] > 0.8]
    if not mid_accs or not edge_accs:
        return False
    return (np.mean(edge_accs) - np.mean(mid_accs)) > 0.5
