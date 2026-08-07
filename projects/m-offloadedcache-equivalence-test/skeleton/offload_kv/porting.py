"""Legacy tuple detection and cache porting utilities."""


def is_legacy_format(cache_obj):
    raise NotImplementedError


def legacy_tuple_to_offloaded_cache(legacy_tuple, device="cpu"):
    raise NotImplementedError


def offloaded_cache_to_legacy_tuple(cache_obj):
    raise NotImplementedError
