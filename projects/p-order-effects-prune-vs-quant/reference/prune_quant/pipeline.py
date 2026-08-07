import numpy as np


def prune_weights(w, sparsity=0.5):
    """Prune weights by absolute magnitude thresholding."""
    if sparsity <= 0.0:
        return w.copy()
    if sparsity >= 1.0:
        return np.zeros_like(w)
    flat_abs = np.abs(w.flatten())
    k = int(np.floor(flat_abs.size * sparsity))
    if k == 0:
        return w.copy()
    threshold = np.partition(flat_abs, k - 1)[k - 1]
    mask = np.abs(w) > threshold
    return w * mask


def quantize_weights(w, num_bits=4, preserve_zero=False):
    """Quantize floating point weights to integer range."""
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    if preserve_zero:
        max_abs = np.max(np.abs(w))
        if max_abs == 0.0:
            return w.copy()
        scale = max_abs / float(qmax)
        q = np.clip(np.round(w / scale), qmin, qmax)
        return q * scale
    else:
        w_min, w_max = np.min(w), np.max(w)
        if w_min == w_max:
            return w.copy()
        scale = (w_max - w_min) / float(qmax - qmin)
        q = np.clip(np.round((w - w_min) / scale) + qmin, qmin, qmax)
        return (q - qmin) * scale + w_min


def measure_both_orders(w, sparsity=0.5, num_bits=4):
    """Measure MSE for Prune-then-Quantize vs Quantize-then-Prune."""
    w_p = prune_weights(w, sparsity)
    w_pq = quantize_weights(w_p, num_bits, preserve_zero=False)
    mse_pq = float(np.mean((w - w_pq) ** 2))

    w_q = quantize_weights(w, num_bits, preserve_zero=False)
    w_qp = prune_weights(w_q, sparsity)
    mse_qp = float(np.mean((w - w_qp) ** 2))

    return {
        "mse_pq": mse_pq,
        "mse_qp": mse_qp,
        "w_pq": w_pq,
        "w_qp": w_qp,
    }


def analyze_interaction(w, sparsity=0.5, num_bits=4):
    """Analyze zero-drift caused by unaligned quantization of pruned weights."""
    w_p = prune_weights(w, sparsity)
    w_pq = quantize_weights(w_p, num_bits, preserve_zero=False)
    zeros_in_p = w_p == 0.0
    zero_drift = float(np.mean((w_pq != 0.0) & zeros_in_p))
    return {
        "zero_drift_pq": zero_drift,
        "range_orig": float(np.ptp(w)),
        "range_pruned": float(np.ptp(w_p)),
        "interaction_detected": 1.0 if zero_drift > 0.0 else 0.0,
    }


def joint_recipe(w, sparsity=0.5, num_bits=4):
    """Apply prune-then-quantize with zero-point preservation."""
    w_p = prune_weights(w, sparsity)
    w_joint = quantize_weights(w_p, num_bits, preserve_zero=True)
    mse_joint = float(np.mean((w - w_joint) ** 2))
    zeros_p = np.sum(w_p == 0.0)
    zeros_joint = np.sum(w_joint == 0.0)
    preserved = 1.0 if zeros_p == zeros_joint else 0.0
    return {
        "w_joint": w_joint,
        "mse_joint": mse_joint,
        "zeros_preserved": preserved,
    }


def measure_gains(w, sparsity=0.5, num_bits=4):
    """Compute compression ratio and throughput speedup factor."""
    dense_bits = 32.0
    compressed_bits = (1.0 - sparsity) * float(num_bits) + 1.0
    compression_ratio = dense_bits / compressed_bits
    effective_bits = float(num_bits) / 16.0
    speedup = 1.0 / ((1.0 - sparsity) * effective_bits)
    return {
        "compression_ratio": float(compression_ratio),
        "speedup_ratio": float(speedup),
    }


def justify_order(layers_dict, sparsity=0.5, num_bits=4):
    """Evaluate pruning-quantization ordering across layer dictionary."""
    mses_pq = []
    mses_qp = []
    wins_pq = 0
    for key, w in layers_dict.items():
        w_p = prune_weights(w, sparsity)
        w_pq = quantize_weights(w_p, num_bits, preserve_zero=True)
        mse_pq = float(np.mean((w - w_pq) ** 2))

        w_q = quantize_weights(w, num_bits, preserve_zero=True)
        w_qp = prune_weights(w_q, sparsity)
        mse_qp = float(np.mean((w - w_qp) ** 2))

        mses_pq.append(mse_pq)
        mses_qp.append(mse_qp)
        if mse_pq < mse_qp:
            wins_pq += 1

    mean_pq = float(np.mean(mses_pq))
    mean_qp = float(np.mean(mses_qp))
    n = len(layers_dict)
    chosen = "prune_first" if mean_pq <= mean_qp else "quant_first"
    win_rate = float(wins_pq) / float(n) if n > 0 else 0.0
    delta = float(mean_qp - mean_pq)
    return {
        "chosen_order": chosen,
        "sample_size": n,
        "win_rate": win_rate,
        "delta": delta,
    }


def transfer_recipe(layers_dict, sparsity=0.5, num_bits=4):
    """Transfer the joint compression recipe across model layers."""
    compressed = {}
    mses = {}
    all_preserved = True
    for key, w in layers_dict.items():
        res = joint_recipe(w, sparsity, num_bits)
        compressed[key] = res["w_joint"]
        mses[key] = res["mse_joint"]
        if res["zeros_preserved"] < 1.0:
            all_preserved = False
    return {
        "compressed_layers": compressed,
        "layer_mses": mses,
        "all_zeros_preserved": 1.0 if all_preserved else 0.0,
    }
