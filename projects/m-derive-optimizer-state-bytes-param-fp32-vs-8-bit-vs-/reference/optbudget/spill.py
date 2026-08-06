import math
from optbudget.bytes import derive_optimizer_bytes_per_param


def derive_spill_trigger_step(cfg):
    vram_limit = cfg["vram_limit_bytes"]
    base_model_bytes = cfg["total_params"] * (cfg["base_precision_bits"] / 8.0)
    activations_bytes = cfg["activation_bytes"]
    gradients_bytes = cfg["gradient_bytes"]
    lora_bytes = cfg["trainable_params"] * (cfg["lora_precision_bits"] / 8.0)
    static_footprint = base_model_bytes + lora_bytes + activations_bytes + gradients_bytes
    if static_footprint >= vram_limit:
        return 0
    remaining_vram = vram_limit - static_footprint
    opt_per_param = derive_optimizer_bytes_per_param(cfg["optimizer_type"])
    trainable = cfg["trainable_params"]
    total_opt_bytes = trainable * opt_per_param
    if total_opt_bytes <= remaining_vram:
        return -1
    excess_bytes = total_opt_bytes - remaining_vram
    base_step = cfg["base_spill_step"]
    growth = cfg["step_growth_rate"]
    steps = base_step + int(math.ceil(excess_bytes / (1048576 * growth)))
    return int(steps)
