def analyze_engine(raw_json_path, simp_json_path):
    """
    Returns a tuple: (device_memory_bytes, raw_layer_count, simp_layer_count)
    """
    raise NotImplementedError


def best_single_change(profile_path, candidates):
    """
    Returns the integer index of the candidate that results in the lowest total network time.
    """
    raise NotImplementedError
