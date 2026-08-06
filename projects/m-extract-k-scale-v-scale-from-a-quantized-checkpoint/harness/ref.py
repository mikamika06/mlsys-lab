import numpy as np

def generate_checkpoint():
    np.random.seed(42)
    return {
        "model.layers.0.self_attn.k_scale": np.random.randn(8).astype(np.float32),
        "model.layers.0.self_attn.v_scale": np.random.randn(8).astype(np.float32),
        "model.layers.1.self_attn.k_scale": np.random.randn(8).astype(np.float32),
        "model.layers.1.self_attn.v_scale": np.random.randn(8).astype(np.float32),
    }

def generate_activations():
    np.random.seed(42)
    return np.random.randn(64, 128).astype(np.float32).tolist()

def extract_scales(checkpoint):
    scales = {}
    for k, v in checkpoint.items():
        if "k_scale" in k or "v_scale" in k:
            scales[k] = np.array(v, dtype=np.float32)
    return scales

def absmax_calibrate(activations):
    arr = np.array(activations, dtype=np.float32)
    max_val = np.max(np.abs(arr))
    if max_val == 0.0:
        return 1.0
    return float(max_val / 127.0)

def sample_count_sweep(activations, counts):
    arr = np.array(activations, dtype=np.float32)
    results = {}
    for c in counts:
        sub = arr[:c] if len(arr) >= c else arr
        results[c] = absmax_calibrate(sub)
    return results
