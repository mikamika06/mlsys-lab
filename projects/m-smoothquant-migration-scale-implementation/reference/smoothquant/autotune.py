import numpy as np
from smoothquant.scale import apply_smoothquant, compute_migration_scales


def quantize_int8(tensor, axis=None):
    """Symmetric INT8 quantization and dequantization simulator."""
    arr = np.asarray(tensor, dtype=np.float32)
    if axis is None:
        max_val = np.max(np.abs(arr))
        scale = max_val / 127.0 if max_val > 1e-8 else 1.0
    else:
        max_val = np.max(np.abs(arr), axis=axis, keepdims=True)
        scale = np.where(max_val > 1e-8, max_val / 127.0, 1.0)

    q = np.clip(np.round(arr / scale), -128, 127)
    return q * scale


def sweep_alpha_per_layer(layer_activations, layer_weights, alpha_candidates):
    """Sweep candidate alpha values per layer to minimize post-quantization output MSE."""
    results = {}
    for layer_name, X in layer_activations.items():
        W = layer_weights[layer_name]
        ref_out = X @ W.T

        act_max = np.max(np.abs(X), axis=0)
        weight_max = np.max(np.abs(W), axis=0)

        best_alpha = None
        best_mse = float("inf")
        best_scales = None

        for alpha in alpha_candidates:
            s = compute_migration_scales(act_max, weight_max, alpha)
            X_s, W_s = apply_smoothquant(X, W, s)

            X_q = quantize_int8(X_s, axis=None)
            W_q = quantize_int8(W_s, axis=1)

            out_q = X_q @ W_q.T
            mse = float(np.mean((ref_out - out_q) ** 2))

            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_scales = s

        results[layer_name] = {
            "alpha": best_alpha,
            "mse": best_mse,
            "scales": best_scales,
        }
    return results
