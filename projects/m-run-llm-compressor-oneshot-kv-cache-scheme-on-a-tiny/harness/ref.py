import numpy as np

CONFIGS = [
    {"layers": 4, "hidden_dim": 64, "num_heads": 4},
    {"layers": 8, "hidden_dim": 128, "num_heads": 8},
]

def generate_calibration_data(config, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = 10
    seq_len = 32
    hidden = config["hidden_dim"]
    data = []
    for _ in range(n_samples):
        act = rng.standard_normal((seq_len, hidden)).astype(np.float32) * 2.5
        data.append(act)
    return data

def compute_reference_scales(activations):
    scales = []
    for act in activations:
        mx = float(np.max(np.abs(act)))
        scale = mx / 448.0 if mx > 0 else 1.0
        scales.append(scale)
    return {"scale": float(np.mean(scales)), "max_val": float(np.max([np.max(np.abs(a)) for a in activations]))}

def repair_recipe(recipe, activations):
    fixed = {}
    for k, v in recipe.items():
        if v == 1.0 or v is None:
            mx = float(np.max(np.abs(activations)))
            fixed[k] = mx / 448.0 if mx > 0 else 1.0
        else:
            fixed[k] = v
    return fixed

def compute_rel_err(ref_out, test_out):
    diff = np.linalg.norm(ref_out - test_out)
    norm = np.linalg.norm(ref_out)
    if norm == 0:
        return 0.0
    return float(diff / norm)
