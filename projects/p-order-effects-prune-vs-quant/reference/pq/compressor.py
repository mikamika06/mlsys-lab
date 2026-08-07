import numpy as np


def prune_weights(w, sparsity):
    flat = np.abs(w.flatten())
    thresh = np.percentile(flat, sparsity * 100)
    mask = np.abs(w) >= thresh
    return w * mask, mask


def quantize_weights(w, bits):
    levels = 2 ** bits - 1
    w_min = np.min(w)
    w_max = np.max(w)
    if w_max == w_min:
        return w, w_min, 1.0
    scale = (w_max - w_min) / levels
    q = np.round((w - w_min) / scale)
    q_clip = np.clip(q, 0, levels)
    dequant = q_clip * scale + w_min
    return dequant, w_min, scale


def evaluate_pipeline(w, x, sparsity, bits, order):
    if order == "p_then_q":
        pw, mask = prune_weights(w, sparsity)
        qw, _, _ = quantize_weights(pw, bits)
        res = np.dot(x, qw.T)
    elif order == "q_then_p":
        qw, _, _ = quantize_weights(w, bits)
        pw, mask = prune_weights(qw, sparsity)
        res = np.dot(x, pw.T)
    else:
        raise ValueError("unknown order")
    err = np.mean((np.dot(x, w.T) - res) ** 2)
    return res, err


def find_joint_recipe(w, x, target_sparsity, target_bits):
    _, err_pq = evaluate_pipeline(w, x, target_sparsity, target_bits, "p_then_q")
    _, err_qp = evaluate_pipeline(w, x, target_sparsity, target_bits, "q_then_p")
    best_order = "p_then_q" if err_pq <= err_qp else "q_then_p"
    return {"order": best_order, "sparsity": target_sparsity, "bits": target_bits, "error": min(err_pq, err_qp)}


def measure_gains(w, sparsity, bits, order):
    _, err = evaluate_pipeline(w, np.random.randn(10, w.shape[1]), sparsity, bits, order)
    size_bytes = (w.size * bits / 8) * (1.0 - sparsity)
    speed_est = 1.0 / (1.0 - 0.5 * sparsity)
    return {"size_bytes": float(size_bytes), "speed_est": float(speed_est), "error": float(err)}


def justify_order(w, x, sparsity, bits):
    _, e1 = evaluate_pipeline(w, x, sparsity, bits, "p_then_q")
    _, e2 = evaluate_pipeline(w, x, sparsity, bits, "q_then_p")
    return {"p_then_q_error": float(e1), "q_then_p_error": float(e2), "recommended": "p_then_q" if e1 <= e2 else "q_then_p"}
