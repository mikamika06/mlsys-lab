def validate_cache(cache_files, available_files):
    return set(cache_files).issubset(set(available_files))


def compute_cold_start_cost(is_cached):
    return 1.0 if is_cached else 100.0
