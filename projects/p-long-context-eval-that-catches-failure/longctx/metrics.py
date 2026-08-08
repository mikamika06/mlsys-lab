import numpy as np

def compute_positional_curve(results):
    sorted_res = sorted(results, key=lambda x: x["needle_index"])
    positions = [r["needle_index"] / max(1, r["context_len"]) for r in sorted_res]
    scores = [r["score"] for r in sorted_res]
    dip = float(np.min(scores) < 0.8 * np.mean(scores))
    return {"positions": positions, "scores": scores, "dip_detected": dip}
