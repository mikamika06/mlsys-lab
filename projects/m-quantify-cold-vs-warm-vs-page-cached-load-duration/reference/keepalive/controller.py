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
