"""Throughput profiler and estimator for preflight validation."""


def estimate_throughput(model_config, quant_config, batch_size, seq_len):
    """Estimate tokens per second throughput for a given configuration."""
    raise NotImplementedError


def compare_throughput_to_fp16(model_config, quant_config, batch_size, seq_len):
    """Compare quantized throughput ratio relative to FP16 execution."""
    raise NotImplementedError
