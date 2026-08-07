def kv_bytes_per_token(config):
    raise NotImplementedError

def effective_capacity(total_bytes, block_size, num_blocks):
    raise NotImplementedError

def peak_headroom_capacity(base_capacity, burst_factor):
    raise NotImplementedError

def predict_trace_concurrency(config, trace, memory_limit):
    raise NotImplementedError

def quantization_breakeven_point(config, precision_bits_high, precision_bits_low, overhead_bytes):
    raise NotImplementedError

def calculate_concurrency(config, workload_spec):
    raise NotImplementedError
