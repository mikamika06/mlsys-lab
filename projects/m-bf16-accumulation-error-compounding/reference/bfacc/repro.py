import numpy as np


def deterministic_sum(chunks, axis=0):
    converted = [np.array(c, dtype=np.float32) for c in chunks]
    stacked = np.stack(converted, axis=0)
    sorted_chunks = np.sort(stacked, axis=0)
    res = np.sum(sorted_chunks, axis=axis, dtype=np.float64)
    return res.astype(np.float32)


def diagnose_loss_spike(loss_history, threshold=2.0):
    losses = np.array(loss_history, dtype=np.float64)
    if len(losses) < 2:
        return {"spiked": False, "step": -1, "ratio": 1.0}
    
    ratios = losses[1:] / np.maximum(losses[:-1], 1e-12)
    max_idx = int(np.argmax(ratios))
    max_ratio = float(ratios[max_idx])
    
    if max_ratio >= threshold:
        return {"spiked": True, "step": max_idx + 1, "ratio": max_ratio}
    return {"spiked": False, "step": -1, "ratio": max_ratio}
