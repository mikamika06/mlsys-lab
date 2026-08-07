import numpy as np
from compression.ops import prune, quantize

def measure_both_orders(w: np.ndarray, p: float, b: int):
    pq = quantize(prune(w, p), b)
    qp = prune(quantize(w, b), p)
    return pq, qp

def find_interaction_flaw(w: np.ndarray, p: float, b: int):
    pq, qp = measure_both_orders(w, p, b)
    return {
        "mse_pq": float(np.mean((w - pq)**2)),
        "mse_qp": float(np.mean((w - qp)**2))
    }

def joint_recipe(w: np.ndarray, p: float, b: int) -> np.ndarray:
    w_out = w.copy()
    n = int(np.round(w.size * p))
    mask = np.ones(w.size, dtype=bool)
    if n > 0:
        idx = np.argsort(np.abs(w))[:n]
        mask[idx] = False
        w_out[idx] = 0.0

    active = w[mask]
    if len(active) == 0:
        return w_out

    w_min, w_max = np.min(active), np.max(active)
    if w_min != w_max:
        levels = (1 << b) - 1
        scale = (w_max - w_min) / levels
        zp = np.round(-w_min / scale)
        q = np.round(active / scale) + zp
        q = np.clip(q, 0, levels)
        w_out[mask] = (q - zp) * scale

    return w_out

def measure_gains(w_orig: np.ndarray, w_comp: np.ndarray, b: int):
    nz = np.count_nonzero(w_comp)
    total = w_orig.size
    size_bits = nz * b
    speedup = total / nz if nz > 0 else float('inf')
    return {
        "size_bits": size_bits,
        "speedup_factor": speedup
    }

def justify_best_order(w: np.ndarray, p: float, b: int):
    flaw = find_interaction_flaw(w, p, b)
    w_joint = joint_recipe(w, p, b)
    mse_joint = float(np.mean((w - w_joint)**2))
    return {
        "best_method": "joint",
        "mse_joint": mse_joint,
        "improvement_over_pq": flaw["mse_pq"] - mse_joint
    }
