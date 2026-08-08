import numpy as np


def analytical_error_comparison(weights, sparsity, q_bits):
    w = np.array(weights, dtype=np.float64)
    flat = np.abs(w.flatten())
    thresh = np.percentile(flat, sparsity * 100.0)
    mask = flat >= thresh
    mask = mask.reshape(w.shape)

    scale = (2.0 ** (q_bits - 1) - 1.0)
    if scale <= 0:
        scale = 1.0

    w_pruned = np.where(mask, w, 0.0)
    w_q_then_p = np.where(mask, np.round(w * scale) / scale, 0.0)

    q_scale = np.round(w * scale) / scale
    w_p_then_q = np.where(mask, q_scale, 0.0)

    err_ptq = np.mean((w - w_pruned)**2) + np.mean((w_pruned - w_p_then_q)**2)
    err_qtp = np.mean((w - q_scale)**2) + np.mean((q_scale - w_q_then_p)**2)

    return {
        "prune_then_quantize_error": float(err_ptq),
        "quantize_then_prune_error": float(err_qtp),
        "delta": float(err_ptq - err_qtp)
    }
