def enable_timing_cache(config):
    config["timing_cache_enabled"] = True
    config["serialization_allowed"] = True
    return config
