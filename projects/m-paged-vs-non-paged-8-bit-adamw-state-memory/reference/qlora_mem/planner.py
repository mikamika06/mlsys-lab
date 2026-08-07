import math
from qlora_mem.adamw import compute_adamw_state_bytes


def qlora_peak_memory_plan(config: dict) -> dict:
    base_params = config["base_params"]
    lora_params = config["lora_params"]
    max_layer_base = config.get("max_layer_base_params", base_params)
    max_layer_lora = config.get("max_layer_lora_params", lora_params)
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    paged_adamw = config.get("paged_adamw", False)
    grad_ckpt = config.get("gradient_checkpointing", False)
    vram_gb = config["vram_gb"]

    base_weight_bytes = math.ceil(base_params * 0.5) + math.ceil(base_params / 64) * 2
    lora_weight_bytes = lora_params * 2
    gradient_bytes = lora_params * 2
    optimizer_bytes = compute_adamw_state_bytes(lora_params, block_size=256, paged=paged_adamw, max_layer_params=max_layer_lora)

    per_layer_act = batch_size * seq_len * hidden_dim * 20
    if grad_ckpt:
        activation_bytes = (num_layers * batch_size * seq_len * hidden_dim * 2) + per_layer_act
    else:
        activation_bytes = num_layers * per_layer_act

    workspace_bytes = max_layer_base * 2

    peak_vram_bytes = (
        base_weight_bytes + lora_weight_bytes + gradient_bytes +
        optimizer_bytes + activation_bytes + workspace_bytes
    )
    limit_bytes = int(vram_gb * (1024 ** 3))
    fits_in_vram = bool(peak_vram_bytes <= limit_bytes)

    return {
        "base_weight_bytes": base_weight_bytes,
        "lora_weight_bytes": lora_weight_bytes,
        "gradient_bytes": gradient_bytes,
        "optimizer_bytes": optimizer_bytes,
        "activation_bytes": activation_bytes,
        "workspace_bytes": workspace_bytes,
        "peak_vram_bytes": peak_vram_bytes,
        "fits_in_vram": fits_in_vram,
    }
