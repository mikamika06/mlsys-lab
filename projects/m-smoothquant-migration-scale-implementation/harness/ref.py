import numpy as np


def generate_synthetic_model_data(seed=42):
    np.random.seed(seed)
    layers = ["layer_attn_q", "layer_attn_v", "layer_mlp_fc1"]
    activations = {}
    weights = {}

    for i, name in enumerate(layers):
        in_dim = 64
        out_dim = 64
        X = np.random.randn(32, in_dim).astype(np.float32)
        W = np.random.randn(out_dim, in_dim).astype(np.float32)

        outlier_idx = (i * 7) % in_dim
        X[:, outlier_idx] *= (i + 1) * 30.0

        activations[name] = X
        weights[name] = W

    return activations, weights


def ref_compute_migration_scales(act_max, weight_max, alpha):
    act_m = np.maximum(np.asarray(act_max, dtype=np.float32), 1e-5)
    weight_m = np.maximum(np.asarray(weight_max, dtype=np.float32), 1e-5)
    scales = np.power(act_m, alpha) / np.power(weight_m, 1.0 - alpha)
    return np.maximum(scales, 1e-5)


def ref_apply_smoothquant(activation, weight, scales):
    s = np.asarray(scales, dtype=np.float32)
    scaled_act = activation / s
    scaled_weight = weight * s
    return scaled_act, scaled_weight


def ref_quantize_int8(tensor, axis=None):
    arr = np.asarray(tensor, dtype=np.float32)
    if axis is None:
        max_val = np.max(np.abs(arr))
        scale = max_val / 127.0 if max_val > 1e-8 else 1.0
    else:
        max_val = np.max(np.abs(arr), axis=axis, keepdims=True)
        scale = np.where(max_val > 1e-8, max_val / 127.0, 1.0)

    q = np.clip(np.round(arr / scale), -128, 127)
    return q * scale


def ref_sweep_alpha_per_layer(layer_activations, layer_weights, alpha_candidates):
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
            s = ref_compute_migration_scales(act_max, weight_max, alpha)
            X_s, W_s = ref_apply_smoothquant(X, W, s)

            X_q = ref_quantize_int8(X_s, axis=None)
            W_q = ref_quantize_int8(W_s, axis=1)

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
