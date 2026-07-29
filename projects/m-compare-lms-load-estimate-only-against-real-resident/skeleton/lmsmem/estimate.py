def param_count(config):
    raise NotImplementedError


def kv_cache_bytes(config, context_length, kv_bytes_per_element=2):
    raise NotImplementedError


def estimate_bytes(config, context_length, bytes_per_param):
    raise NotImplementedError
