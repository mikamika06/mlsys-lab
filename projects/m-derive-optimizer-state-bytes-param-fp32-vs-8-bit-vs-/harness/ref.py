import math


CONFIGS = [
    {
        "trainable_params": 20000000,
        "total_params": 7000000000,
        "base_precision_bits": 16,
        "lora_precision_bits": 16,
        "activation_bytes": 524288000,
        "gradient_bytes": 20971520,
        "optimizer_type": "adam_8bit",
        "vram_limit_bytes": 17179869184,
        "base_spill_step": 100,
        "step_growth_rate": 1.05
    },
    {
        "trainable_params": 50000000,
        "total_params": 13000000000,
        "base_precision_bits": 4,
        "lora_precision_bits": 16,
        "activation_bytes": 1048576000,
        "gradient_bytes": 52428800,
        "optimizer_type": "adam_fp32",
        "vram_limit_bytes": 25769803776,
        "base_spill_step": 50,
        "step_growth_rate": 1.1
    },
    {
        "trainable_params": 10000000,
        "total_params": 3000000000,
        "base_precision_bits": 8,
        "lora_precision_bits": 16,
        "activation_bytes": 262144000,
        "gradient_bytes": 10485760,
        "optimizer_type": "sgd",
        "vram_limit_bytes": 8589934592,
        "base_spill_step": 200,
        "step_growth_rate": 1.02
    }
]


def derive_optimizer_bytes_per_param(opt_type):
    if opt_type == "adam_fp32":
        return 12.0
    elif opt_type == "adam_8bit":
        return 2.0
    elif opt_type == "sgd":
        return 4.0
    else:
        raise ValueError("Unknown optimizer type")


def derive_total_memory_budget(cfg):
    trainable = cfg["trainable_params"]
    total = cfg["total_params"]
    base_bits = cfg["base_precision_bits"]
    lora_bits = cfg["lora_precision_bits"]
    base_model_bytes = total * (base_bits / 8.0)
    lora_adapter_bytes = trainable * (lora_bits / 8.0)
    gradients_bytes = cfg["gradient_bytes"]
    activations_bytes = cfg["activation_bytes"]
    opt_bytes_per_param = derive_optimizer_bytes_per_param(cfg["optimizer_type"])
    optimizer_state_bytes = trainable * opt_bytes_per_param
    total_budget = (
        base_model_bytes
        + lora_adapter_bytes
        + gradients_bytes
        + activations_bytes
        + optimizer_state_bytes
    )
    return float(total_budget)


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
