import numpy as np

CONFIGS = [
    {
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "intermediate_size": 3072,
    },
    {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
    },
    {
        "num_hidden_layers": 24,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "intermediate_size": 8192,
    },
]

MODULE_TREES = [
    [
        f"model.layers.{i}.self_attn.{m}"
        for i in range(12)
        for m in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ]
    + [
        f"model.layers.{i}.mlp.{m}"
        for i in range(12)
        for m in ["gate_proj", "up_proj", "down_proj"]
    ],
    [
        f"transformer.h.{i}.attn.{m}"
        for i in range(32)
        for m in ["q_proj", "v_proj"]
    ],
]


def count_trainable_parameters(model_config, target_modules, r):
    num_layers = model_config["num_hidden_layers"]
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config.get("num_key_value_heads", num_heads)
    intermediate_size = model_config["intermediate_size"]
    head_dim = model_config.get("head_dim", hidden_size // num_heads)

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    dim_map = {
        "q_proj": (hidden_size, q_dim),
        "k_proj": (hidden_size, kv_dim),
        "v_proj": (hidden_size, kv_dim),
        "o_proj": (q_dim, hidden_size),
        "gate_proj": (hidden_size, intermediate_size),
        "up_proj": (hidden_size, intermediate_size),
        "down_proj": (intermediate_size, hidden_size),
    }

    total_per_layer = 0
    for mod in target_modules:
        if mod in dim_map:
            d_in, d_out = dim_map[mod]
            total_per_layer += r * (d_in + d_out)

    return total_per_layer * num_layers


def resolve_target_modules(named_modules, target_shorthands):
    shorthands = set(target_shorthands)
    resolved = []
    for path in named_modules:
        leaf = path.split(".")[-1]
        if leaf in shorthands or path in shorthands:
            resolved.append(path)
    return sorted(resolved)


def compute_scaling_factor(alpha, r, mode="lora"):
    if mode == "lora":
        return alpha / r
    elif mode == "naive":
        return alpha
    else:
        raise ValueError(f"Unknown mode: {mode}")


def apply_lora_scaling(x, weight_a, weight_b, alpha, r, mode="lora"):
    scale = compute_scaling_factor(alpha, r, mode=mode)
    h = np.dot(x, weight_a.T)
    delta = np.dot(h, weight_b.T)
    return scale * delta
