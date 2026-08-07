def compute_latencies(prefill_len, decode_steps, stateful_params, stateless_params):
    raise NotImplementedError


def peak_memory_delta(cache_shape, dtype_bytes):
    raise NotImplementedError
