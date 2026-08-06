import numpy as np

def diagnose_divergence(loss_history, grad_norms):
    losses = np.array(loss_history, dtype=np.float64)
    norms = np.array(grad_norms, dtype=np.float64)
    if np.any(np.isnan(losses)) or np.any(np.isnan(norms)):
        nan_idx = int(np.where(np.isnan(losses) | np.isnan(norms))[0][0])
        return {"diverged": True, "step": nan_idx, "reason": "nan_detected"}
    if len(losses) > 10:
        rolling_mean = np.convolve(losses, np.ones(5)/5, mode="valid")
        if np.any(rolling_mean > 1e4 * rolling_mean[0] + 10.0):
            spike_idx = int(np.argmax(losses > 1e4 * losses[0] + 10.0))
            return {"diverged": True, "step": spike_idx, "reason": "loss_spike"}
    max_norm = np.max(norms) if len(norms) > 0 else 0.0
    if max_norm > 1e5:
        norm_idx = int(np.argmax(norms > 1e5))
        return {"diverged": True, "step": norm_idx, "reason": "gradient_explosion"}
    return {"diverged": False, "step": -1, "reason": "stable"}
