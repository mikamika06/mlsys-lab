import numpy as np

MODELS = [
    {
        "lm_head_name": "lm_head",
        "modules": {
            "model.embed_tokens": {"type": "embedding", "num_embeddings": 1000, "embedding_dim": 64},
            "model.layers.0.self_attn.q_proj": {"type": "linear", "in_features": 64, "out_features": 64},
            "model.layers.0.self_attn.k_proj": {"type": "linear", "in_features": 64, "out_features": 64},
            "model.layers.0.self_attn.v_proj": {"type": "linear", "in_features": 64, "out_features": 64},
            "model.layers.0.self_attn.o_proj": {"type": "linear", "in_features": 64, "out_features": 64},
            "model.layers.0.mlp.gate_proj": {"type": "linear", "in_features": 64, "out_features": 256},
            "model.layers.0.mlp.up_proj": {"type": "linear", "in_features": 64, "out_features": 256},
            "model.layers.0.mlp.down_proj": {"type": "linear", "in_features": 256, "out_features": 64},
            "lm_head": {"type": "linear", "in_features": 64, "out_features": 1000},
        }
    },
    {
        "lm_head_name": "head",
        "modules": {
            "backbone.layer.0.attn": {"type": "linear", "in_features": 128, "out_features": 128},
            "backbone.layer.0.mlp": {"type": "linear", "in_features": 128, "out_features": 512},
            "head": {"type": "linear", "in_features": 128, "out_features": 10},
        }
    }
]


def count_trainable_params(model_structure, target_modules, r, use_rslora=False):
    total = 0
    modules = model_structure.get("modules", {})
    targets = set(target_modules) if isinstance(target_modules, (list, tuple, set)) else {target_modules}
    for name, spec in modules.items():
        if spec.get("type") != "linear":
            continue
        if name in targets or any(name.endswith("." + t) for t in targets):
            in_dim = spec["in_features"]
            out_dim = spec["out_features"]
            total += r * (in_dim + out_dim)
    return total


def sweep_ranks(model_structure, target_modules, ranks, use_rslora=False):
    return {r: count_trainable_params(model_structure, target_modules, r, use_rslora=use_rslora) for r in ranks}


def expand_target_modules(model_structure, target_modules="all-linear"):
    modules = model_structure.get("modules", {})
    if target_modules != "all-linear":
        if isinstance(target_modules, str):
            return [target_modules]
        return list(target_modules)
    expanded = []
    lm_head_name = model_structure.get("lm_head_name", "lm_head")
    for name, spec in modules.items():
        if spec.get("type") == "linear":
            if name == lm_head_name or name.endswith("." + lm_head_name):
                continue
            expanded.append(name)
    return sorted(expanded)


def measure_dropout_stochasticity(x, w_a, w_b, lora_alpha, lora_dropout, num_samples, seed=42):
    rng = np.random.RandomState(seed)
    r = w_a.shape[0]
    scaling = lora_alpha / r
    outputs = []
    for _ in range(num_samples):
        if lora_dropout > 0.0:
            mask = (rng.uniform(0.0, 1.0, size=x.shape) >= lora_dropout).astype(np.float64)
            x_eff = (x * mask) / (1.0 - lora_dropout)
        else:
            x_eff = x.astype(np.float64)
        h = np.dot(x_eff, w_a.T)
        out = np.dot(h, w_b.T) * scaling
        outputs.append(out)
    stacked = np.stack(outputs, axis=0)
    var = np.var(stacked, axis=0)
    mean_var = float(np.mean(var))
    is_stochastic = mean_var > 1e-9
    return {
        "mean_variance": mean_var,
        "is_stochastic": is_stochastic,
        "sample_outputs": outputs
    }
