def compute_relative_error(fp32_output, quantized_output):
    raise NotImplementedError


def compute_ir_size_reduction(fp32_size_bytes, int8_size_bytes):
    raise NotImplementedError


def compute_benchmark_latency_gain(fp32_latency_ms, int8_latency_ms):
    raise NotImplementedError
