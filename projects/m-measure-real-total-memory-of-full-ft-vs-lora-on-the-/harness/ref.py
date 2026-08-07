def get_layer_shapes(config: dict) -> dict[str, tuple[int, int]]:
    h = config["hidden_size"]
    inter = config["intermediate_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = h // n_heads
    kv_dim = n_kv_heads * head_dim
    return {
        "q_proj": (h, h),
        "k_proj": (h, kv_dim),
        "v_proj": (h, kv_dim),
        "o_proj": (h, h),
        "gate_proj": (h, inter),
        "up_proj": (h, inter),
        "down_proj": (inter, h),
    }


def count_base_params(config: dict) -> int:
    h = config["hidden_size"]
    v = config["vocab_size"]
    layers = config["num_hidden_layers"]
    shapes = get_layer_shapes(config)
    attn = sum(
        din * dout
        for name, (din, dout) in shapes.items()
        if name in ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    mlp = sum(
        din * dout
        for name, (din, dout) in shapes.items()
        if name in ("gate_proj", "up_proj", "down_proj")
    )
    layer_total = attn + mlp + 2 * h
    embeddings = v * h
    final_norm = h
    lm_head = v * h
    return embeddings + layers * layer_total + final_norm + lm_head


def count_lora_params(config: dict, lora_config: dict) -> int:
    r = lora_config["r"]
    targets = lora_config["target_modules"]
    shapes = get_layer_shapes(config)
    layers = config["num_hidden_layers"]
    per_layer = sum(
        r * (din + dout)
        for m in targets
        if m in shapes
        for din, dout in [shapes[m]]
    )
    return layers * per_layer


def count_trainable_params(config: dict, lora_config: dict | None = None) -> int:
    if lora_config is None:
        return count_base_params(config)
    return count_lora_params(config, lora_config)


def estimate_activation_memory(
    config: dict,
    batch_size: int = 1,
    seq_len: int = 512,
    activation_checkpointing: bool = False,
) -> int:
    h = config["hidden_size"]
    inter = config["intermediate_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = h // n_heads
    kv_dim = n_kv_heads * head_dim
    layers = config["num_hidden_layers"]

    act_elements = 10 * h + 2 * kv_dim + 5 * inter + 2 * n_heads * seq_len
    act_bytes_per_layer = batch_size * seq_len * act_elements * 2

    if activation_checkpointing:
        return int(round(2 * act_bytes_per_layer))
    return int(round(layers * act_bytes_per_layer))


def estimate_memory_footprint(
    config: dict,
    mode: str,
    lora_config: dict | None = None,
    batch_size: int = 1,
    seq_len: int = 512,
    activation_checkpointing: bool = False,
) -> dict:
    base_p = count_base_params(config)

    if mode == "full_ft":
        base_bytes = round(2.0 * base_p)
        lora_bytes = 0
        grad_bytes = round(2.0 * base_p)
        opt_bytes = round(12.0 * base_p)
        trainable_p = base_p
    elif mode == "lora_bf16":
        lora_p = count_lora_params(config, lora_config) if lora_config else 0
        base_bytes = round(2.0 * base_p)
        lora_bytes = round(2.0 * lora_p)
        grad_bytes = round(2.0 * lora_p)
        opt_bytes = round(12.0 * lora_p)
        trainable_p = lora_p
    elif mode == "qlora_4bit":
        lora_p = count_lora_params(config, lora_config) if lora_config else 0
        base_bytes = round(0.55 * base_p)
        lora_bytes = round(2.0 * lora_p)
        grad_bytes = round(2.0 * lora_p)
        opt_bytes = round(12.0 * lora_p)
        trainable_p = lora_p
    else:
        raise ValueError(f"Unknown mode: {mode}")

    act_bytes = estimate_activation_memory(
        config, batch_size, seq_len, activation_checkpointing
    )
    static_bytes = int(base_bytes + lora_bytes + grad_bytes + opt_bytes)
    total_peak = static_bytes + act_bytes

    return {
        "base_weights_bytes": int(base_bytes),
        "lora_weights_bytes": int(lora_bytes),
        "gradients_bytes": int(grad_bytes),
        "optimizer_bytes": int(opt_bytes),
        "activations_bytes": int(act_bytes),
        "total_static_bytes": int(static_bytes),
        "total_peak_bytes": int(total_peak),
        "trainable_params": int(trainable_p),
    }


CONFIGS = [
    {
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 16000,
    },
    {
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 12,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "vocab_size": 32000,
    },
    {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
    },
]

LORA_CONFIGS = [
    {"r": 8, "target_modules": ["q_proj", "v_proj"]},
    {"r": 16, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    {
        "r": 32,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
]
