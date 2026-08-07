import numpy as np

def evaluate_mse(w, w_pruned, bias, x):
    y_target = w @ x
    y_pred = w_pruned @ x + bias[:, None]
    return float(np.mean((y_target - y_pred)**2))

def compare_methods(w, x, sparsity=0.5):
    from prune.importance import score_magnitude, score_wanda
    from prune.layer import prune_unstructured, correct_bias

    s_mag = score_magnitude(w)
    w_mag, _ = prune_unstructured(w, s_mag, sparsity)
    bias_zero = np.zeros(w.shape[0])
    mse_mag = evaluate_mse(w, w_mag, bias_zero, x)

    s_wan = score_wanda(w, x)
    w_wan, _ = prune_unstructured(w, s_wan, sparsity)
    bias_wan = correct_bias(w, w_wan, x)
    mse_wan = evaluate_mse(w, w_wan, bias_wan, x)

    return float(mse_mag), float(mse_wan)
