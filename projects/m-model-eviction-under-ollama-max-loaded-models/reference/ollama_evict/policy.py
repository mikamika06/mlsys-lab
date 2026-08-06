def select_evict(loaded_models, access_times, max_loaded):
    if len(loaded_models) <= max_loaded:
        return None
    sorted_models = sorted(loaded_models, key=lambda m: access_times.get(m, 0))
    return sorted_models[0]
