import numpy as np


def select_scaling_mode(tensor, history, static_threshold=0.1):
    arr = np.array(tensor, dtype=np.float32)
    if not history:
        return "dynamic", float(np.max(np.abs(arr)))
    variance = np.var(arr)
    mean_hist_var = np.mean([np.var(np.array(h, dtype=np.float32)) for h in history])
    if mean_hist_var == 0:
        var_ratio = 1.0
    else:
        var_ratio = abs(variance - mean_hist_var) / mean_hist_var
    if var_ratio < static_threshold:
        static_scale = float(np.max([np.max(np.abs(h)) for h in history]))
        return "static", static_scale
    return "dynamic", float(np.max(np.abs(arr)))
