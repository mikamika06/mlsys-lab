from ftmem.lora import count_lora_params
from ftmem.model import count_base_params


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
