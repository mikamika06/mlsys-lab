def estimate_load_duration(model_size_mb, state, disk_bw=500.0, cache_bw=4000.0, overhead=0.05):
    if state == "warm":
        return 0.0
    if state == "page_cached":
        return overhead + (model_size_mb / cache_bw)
    if state == "cold":
        return overhead + (model_size_mb / disk_bw)
    raise ValueError(f"unknown state: {state}")

class KeepAliveController:
    def __init__(self, memory_cap_mb):
        self.memory_cap_mb = memory_cap_mb
        self.loaded_models = {}

    def update(self, current_models):
        current_memory = sum(m["size_mb"] for m in current_models if m["id"] in self.loaded_models)
        new_loaded = {}
        for m in sorted(current_models, key=lambda x: x["priority"], reverse=True):
            mid = m["id"]
            size = m["size_mb"]
            if mid in self.loaded_models:
                new_loaded[mid] = m
                continue
            if current_memory + size <= self.memory_cap_mb:
                new_loaded[mid] = m
                current_memory += size
        self.loaded_models = new_loaded
        return list(self.loaded_models.keys())

def select_keep_alive(models, memory_cap_mb, request_frequencies):
    scored = []
    for m in models:
        mid = m["id"]
        freq = request_frequencies.get(mid, 0.0)
        score = freq * m["size_mb"]
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    current_mem = 0.0
    for _, m in scored:
        if current_mem + m["size_mb"] <= memory_cap_mb:
            selected.append(m["id"])
            current_mem += m["size_mb"]
    return selected
