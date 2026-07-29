import math

from .estimate import param_count, kv_cache_bytes

PAGE_SIZE = 4096
SCRATCH_TOKENS = 256
ACTIVATION_BYTES_PER_ELEMENT = 4
RUNTIME_OVERHEAD_BYTES = 64 * 1024 * 1024


def _round_up(value, block):
    return -(-value // block) * block


def resident_bytes(config, context_length, bytes_per_param, page_size=PAGE_SIZE):
    weight_bytes = int(math.ceil(param_count(config) * bytes_per_param))
    kv_bytes = kv_cache_bytes(config, context_length)
    scratch = config["n_layers"] * SCRATCH_TOKENS * config["intermediate_size"] * ACTIVATION_BYTES_PER_ELEMENT
    return (_round_up(weight_bytes, page_size) + _round_up(kv_bytes, page_size)
            + scratch + RUNTIME_OVERHEAD_BYTES)


def relative_error(estimate, resident):
    if resident == 0:
        return 0.0
    return abs(resident - estimate) / resident
