MODEL_CONFIGS = [
    {
        "name": "llama-7b",
        "num_params": 7_000_000_000,
        "bytes_per_param": 2,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "context_len": 2048,
    },
    {
        "name": "llama-70b-gqa",
        "num_params": 70_000_000_000,
        "bytes_per_param": 2,
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "context_len": 4096,
    },
    {
        "name": "mistral-7b-gqa",
        "num_params": 7_200_000_000,
        "bytes_per_param": 2,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "context_len": 8192,
    },
    {
        "name": "tiny-model",
        "num_params": 1_000_000_000,
        "bytes_per_param": 2,
        "num_layers": 16,
        "num_heads": 16,
        "num_kv_heads": 16,
        "head_dim": 64,
        "context_len": 512,
    },
    {
        "name": "dense-13b",
        "num_params": 13_000_000_000,
        "bytes_per_param": 2,
        "num_layers": 40,
        "num_heads": 40,
        "num_kv_heads": 40,
        "head_dim": 128,
        "context_len": 1024,
    },
]

HARDWARE_SPECS = [
    {"name": "A100-SXM4-80GB", "peak_flops": 312e12, "peak_bandwidth": 2039e9},
    {"name": "H100-SXM-80GB", "peak_flops": 989e12, "peak_bandwidth": 3350e9},
    {"name": "L40S", "peak_flops": 366e12, "peak_bandwidth": 864e9},
    {"name": "V100-32GB", "peak_flops": 125e12, "peak_bandwidth": 900e9},
    {"name": "A10G", "peak_flops": 125e12, "peak_bandwidth": 600e9},
]

WORKLOAD_PROFILES = [
    {"prefill_batch": 1, "prefill_len": 512, "decode_batch": 1, "decode_len": 512},
    {"prefill_batch": 4, "prefill_len": 2048, "decode_batch": 32, "decode_len": 2048},
    {"prefill_batch": 8, "prefill_len": 4096, "decode_batch": 128, "decode_len": 4096},
    {"prefill_batch": 2, "prefill_len": 128, "decode_batch": 16, "decode_len": 1024},
    {"prefill_batch": 16, "prefill_len": 1024, "decode_batch": 256, "decode_len": 1024},
    {"prefill_batch": 1, "prefill_len": 8192, "decode_batch": 4, "decode_len": 8192},
    {"prefill_batch": 32, "prefill_len": 512, "decode_batch": 512, "decode_len": 512},
    {"prefill_batch": 2, "prefill_len": 2048, "decode_batch": 64, "decode_len": 2048},
    {"prefill_batch": 4, "prefill_len": 1024, "decode_batch": 8, "decode_len": 1024},
    {"prefill_batch": 8, "prefill_len": 256, "decode_batch": 1024, "decode_len": 256},
]

MEASURED_METRICS = [
    {"batch_size": 1, "avg_seq_len": 1024, "tokens_per_second": 45.0, "time_seconds": 10.0},
    {"batch_size": 8, "avg_seq_len": 2048, "tokens_per_second": 320.0, "time_seconds": 10.0},
    {"batch_size": 16, "avg_seq_len": 512, "tokens_per_second": 600.0, "time_seconds": 10.0},
    {"batch_size": 32, "avg_seq_len": 4096, "tokens_per_second": 1100.0, "time_seconds": 10.0},
    {"batch_size": 64, "avg_seq_len": 1024, "tokens_per_second": 1800.0, "time_seconds": 10.0},
    {"batch_size": 4, "avg_seq_len": 8192, "tokens_per_second": 150.0, "time_seconds": 10.0},
    {"batch_size": 128, "avg_seq_len": 256, "tokens_per_second": 3500.0, "time_seconds": 10.0},
    {"batch_size": 2, "avg_seq_len": 1024, "tokens_per_second": 88.0, "time_seconds": 10.0},
    {"batch_size": 256, "avg_seq_len": 512, "tokens_per_second": 5000.0, "time_seconds": 10.0},
    {"batch_size": 16, "avg_seq_len": 2048, "tokens_per_second": 550.0, "time_seconds": 10.0},
]


def find_decode_compute_bound_batch_size(model_config, hardware_specs):
    num_params = model_config["num_params"]
    bytes_per_param = model_config["bytes_per_param"]
    num_layers = model_config["num_layers"]
    num_heads = model_config["num_heads"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]
    context_len = model_config.get("context_len", 1024)

    peak_flops = hardware_specs["peak_flops"]
    peak_bandwidth = hardware_specs["peak_bandwidth"]
    roofline_knee = peak_flops / peak_bandwidth

    weight_bytes_per_token = num_params * bytes_per_param
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param * context_len
    flops_per_token = 2 * num_params

    max_b = 16384
    for b in range(1, max_b + 1):
        total_bytes = weight_bytes_per_token + b * kv_bytes_per_token
        total_flops = b * flops_per_token
        intensity = total_flops / total_bytes
        if intensity >= roofline_knee:
            return b
    return max_b


def calculate_operational_intensity(model_config, batch_size, seq_len, phase="decode"):
    num_params = model_config["num_params"]
    bytes_per_param = model_config["bytes_per_param"]
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]

    if phase == "prefill":
        flops = 2 * num_params * batch_size * seq_len
        bytes_transferred = (num_params * bytes_per_param) + (
            2 * num_layers * num_kv_heads * head_dim * bytes_per_param * batch_size * seq_len
        )
    else:
        flops = 2 * num_params * batch_size
        bytes_transferred = (num_params * bytes_per_param) + (
            2 * num_layers * num_kv_heads * head_dim * bytes_per_param * batch_size * seq_len
        )
    return flops / bytes_transferred


def classify_workload_dominance(workload_profiles, model_config, hardware_specs):
    roofline_knee = hardware_specs["peak_flops"] / hardware_specs["peak_bandwidth"]
    classifications = []

    for profile in workload_profiles:
        prefill_batch = profile["prefill_batch"]
        prefill_len = profile["prefill_len"]
        decode_batch = profile["decode_batch"]
        decode_len = profile["decode_len"]

        intensity_prefill = calculate_operational_intensity(
            model_config, prefill_batch, prefill_len, phase="prefill"
        )
        intensity_decode = calculate_operational_intensity(
            model_config, decode_batch, decode_len, phase="decode"
        )

        if intensity_prefill >= roofline_knee and intensity_decode < roofline_knee:
            classifications.append("prefill_compute_decode_memory")
        elif intensity_prefill >= roofline_knee and intensity_decode >= roofline_knee:
            classifications.append("compute_heavy")
        elif intensity_prefill < roofline_knee and intensity_decode < roofline_knee:
            classifications.append("memory_heavy")
        else:
            classifications.append("prefill_memory_decode_compute")

    return classifications


def extract_achieved_hbm_bandwidth(measured_metrics, model_config):
    num_params = model_config["num_params"]
    bytes_per_param = model_config["bytes_per_param"]
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]

    batch_size = measured_metrics["batch_size"]
    avg_seq_len = measured_metrics["avg_seq_len"]
    tokens_per_second = measured_metrics["tokens_per_second"]
    time_seconds = measured_metrics["time_seconds"]

    weight_bytes = num_params * bytes_per_param
    kv_bytes = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param * batch_size * avg_seq_len
    bytes_per_decode_step = weight_bytes + kv_bytes

    total_tokens = tokens_per_second * time_seconds
    num_decode_steps = total_tokens / batch_size
    total_bytes = bytes_per_decode_step * num_decode_steps

    achieved_bandwidth_gbps = (total_bytes / time_seconds) / 1e9
    return achieved_bandwidth_gbps
