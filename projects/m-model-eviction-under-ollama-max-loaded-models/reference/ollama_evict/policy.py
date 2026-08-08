def select_eviction(loaded_models, max_loaded):
    loaded = [m for m, data in loaded_models.items() if data.get("loaded", True)]
    if len(loaded) <= max_loaded:
        return None
    sorted_models = sorted(loaded, key=lambda name: (loaded_models[name]["last_used"], loaded_models[name]["access_count"]))
    return sorted_models[0]
