"""Throughput profiler and estimator for preflight validation."""

from preflight.validator import BYTES_PER_ELEMENT


def estimate_throughput(model_config, quant_config, batch_size, seq_len):
    """Estimate tokens per second throughput for a given configuration."""
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
    """Compare quantized throughput ratio relative to FP16 execution."""
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
