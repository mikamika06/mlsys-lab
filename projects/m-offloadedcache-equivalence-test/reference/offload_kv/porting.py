"""Legacy tuple detection and cache porting utilities."""

import torch
from offload_kv.cache import OffloadedCache


def is_legacy_format(cache_obj):
    if isinstance(cache_obj, tuple):
        if len(cache_obj) == 0:
            return True
        first = cache_obj[0]
        if isinstance(first, tuple) and len(first) == 2:
            return True
    return False


def legacy_tuple_to_offloaded_cache(legacy_tuple, device="cpu"):
    num_layers = len(legacy_tuple)
    cache = OffloadedCache(num_layers=num_layers, offload_device=device)
    for layer_idx, (k, v) in enumerate(legacy_tuple):
        cache.update(k, v, layer_idx)
    return cache


def offloaded_cache_to_legacy_tuple(cache_obj):
    legacy = []
    for layer_idx in range(cache_obj.num_layers):
        k = cache_obj.key_cache[layer_idx]
        v = cache_obj.value_cache[layer_idx]
        legacy.append((k, v))
    return tuple(legacy)
