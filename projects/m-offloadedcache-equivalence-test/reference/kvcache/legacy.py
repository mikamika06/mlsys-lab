import numpy as np
from .offloaded import DynamicCache

def is_legacy_tuple(past_key_values):
    if not isinstance(past_key_values, tuple):
        return False
    if len(past_key_values) == 0:
        return False
    return all(isinstance(layer, tuple) and len(layer) == 2 for layer in past_key_values)

def port_legacy_to_cache(past_key_values):
    if not is_legacy_tuple(past_key_values):
        raise ValueError("Input is not a legacy tuple")

    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(past_key_values):
        cache.update(k, v, layer_idx)
    return cache
