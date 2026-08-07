"""Reference oracle implementations and test configurations for harness checkers."""

CONFIGS = [
    {
        "model": {
            "num_parameters": 7_000_000_000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_model_len": 4096,
            "max_num_seqs": 16,
        },
        "quant": {"quant_type": "awq", "compute_speedup": 1.4, "memory_bandwidth_gbps": 1500.0},
        "gpu": {"num_gpus": 1, "memory_per_gpu_bytes": 16 * 1024 * 1024 * 1024, "gpu_memory_utilization": 0.90},
        "batch_size": 16,
        "seq_len": 512,
    },
    {
        "model": {
            "num_parameters": 13_000_000_000,
            "hidden_size": 5120,
            "num_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 40,
            "max_model_len": 8192,
            "max_num_seqs": 32,
        },
        "quant": {"quant_type": "fp8", "compute_speedup": 1.8, "memory_bandwidth_gbps": 2000.0},
        "gpu": {"num_gpus": 2, "memory_per_gpu_bytes": 24 * 1024 * 1024 * 1024, "gpu_memory_utilization": 0.85},
        "batch_size": 32,
        "seq_len": 1024,
    },
    {
        "model": {
            "num_parameters": 70_000_000_000,
            "hidden_size": 8192,
            "num_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "max_model_len": 4096,
            "max_num_seqs": 64,
        },
        "quant": {"quant_type": "gptq", "compute_speedup": 1.3, "memory_bandwidth_gbps": 1500.0},
        "gpu": {"num_gpus": 4, "memory_per_gpu_bytes": 40 * 1024 * 1024 * 1024, "gpu_memory_utilization": 0.90},
        "batch_size": 64,
        "seq_len": 2048,
    },
]

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


def estimate_throughput(model_config, quant_config, batch_size, seq_len):
    num_params = model_config["num_parameters"]
    q_type = quant_config.get("quant_type", "fp16").lower()
    bpp = BYTES_PER_ELEMENT.get(q_type, 2.0)

    compute_speedup = quant_config.get("compute_speedup", 1.0)
    memory_bandwidth = quant_config.get("memory_bandwidth_gbps", 1500.0) * 1e9

    weight_transfer_time = (num_params * bpp) / memory_bandwidth
    compute_time = (2 * num_params * batch_size) / (100e12 * compute_speedup)

    step_time = max(weight_transfer_time, compute_time)
    tokens_per_second = (batch_size * seq_len) / (step_time * seq_len)
    return tokens_per_second


def compare_throughput_to_fp16(model_config, quant_config, batch_size, seq_len):
    fp16_config = {"quant_type": "fp16", "compute_speedup": 1.0, "memory_bandwidth_gbps": quant_config.get("memory_bandwidth_gbps", 1500.0)}

    quant_tps = estimate_throughput(model_config, quant_config, batch_size, seq_len)
    fp16_tps = estimate_throughput(model_config, fp16_config, batch_size, seq_len)

    ratio = quant_tps / fp16_tps if fp16_tps > 0 else 0.0
    return {
        "quant_throughput": quant_tps,
        "fp16_throughput": fp16_tps,
        "speedup_ratio": ratio,
        "is_faster": ratio > 1.0,
    }
