"""Preflight validation logic for vLLM deployment configurations."""

BYTES_PER_ELEMENT = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "awq": 0.5,
    "gptq": 0.5,
    "int4": 0.5,
}


def estimate_memory_bytes(model_config, quant_config, num_gpus):
    """Estimate total memory required per GPU in bytes."""
    num_params = model_config["num_parameters"]
    q_type = quant_config.get("quant_type", "fp16").lower()
    bpp = BYTES_PER_ELEMENT.get(q_type, 2.0)

    weights_bytes = (num_params * bpp) / num_gpus

    hidden_size = model_config["hidden_size"]
    num_layers = model_config["num_layers"]
    max_seq_len = model_config.get("max_model_len", 4096)
    max_batch_size = model_config.get("max_num_seqs", 32)

    kv_heads = model_config.get("num_key_value_heads", model_config.get("num_attention_heads"))
    num_heads = model_config["num_attention_heads"]
    head_dim = hidden_size // num_heads

    kv_cache_bytes = (
        2 * num_layers * kv_heads * head_dim * max_seq_len * max_batch_size * 2.0
    ) / num_gpus

    activation_bytes = (
        max_batch_size * max_seq_len * hidden_size * 2.0
    ) / num_gpus

    cuda_overhead = 1024 * 1024 * 1024

    return weights_bytes + kv_cache_bytes + activation_bytes + cuda_overhead


def validate_fit(model_config, quant_config, gpu_specs):
    """Validate if quantized model fits on given GPU configuration."""
    num_gpus = gpu_specs["num_gpus"]
    memory_per_gpu = gpu_specs["memory_per_gpu_bytes"]
    utilization_limit = gpu_specs.get("gpu_memory_utilization", 0.90)

    est_bytes = estimate_memory_bytes(model_config, quant_config, num_gpus)
    max_allowed_bytes = memory_per_gpu * utilization_limit

    fits = est_bytes <= max_allowed_bytes
    return {
        "fits": fits,
        "estimated_bytes_per_gpu": est_bytes,
        "max_allowed_bytes_per_gpu": max_allowed_bytes,
        "headroom_bytes": max_allowed_bytes - est_bytes,
    }
