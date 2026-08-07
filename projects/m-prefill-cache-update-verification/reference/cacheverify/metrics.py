import numpy as np


def compute_latencies(prefill_len, decode_steps, stateful_params, stateless_params):
    stateful_prefill = stateful_params["base_overhead"] + prefill_len * stateful_params["per_token"]
    stateful_decode = decode_steps * stateful_params["decode_per_token"]
    stateless_prefill = stateless_params["base_overhead"] + prefill_len * stateless_params["per_token"]
    stateless_decode = decode_steps * (stateless_params["decode_per_token"] + stateless_params["io_penalty"])
    return {
        "stateful_total": float(stateful_prefill + stateful_decode),
        "stateless_total": float(stateless_prefill + stateless_decode)
    }


def peak_memory_delta(cache_shape, dtype_bytes):
    total_elements = int(np.prod(cache_shape))
    return float(total_elements * dtype_bytes)
