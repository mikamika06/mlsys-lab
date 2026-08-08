from ollama_evict.tracker import ModelTracker
from ollama_evict.policy import select_evict


class ModelManager:
    def __init__(self, max_loaded):
        self.max_loaded = max_loaded
        self.loaded = []
        self.tracker = ModelTracker()
        self.time = 0

    def request(self, model_name):
        self.time += 1
        self.tracker.touch(model_name, self.time)
        if model_name not in self.loaded:
            self.loaded.append(model_name)
        evicted = []
        while len(self.loaded) > self.max_loaded:
            victim = select_evict(self.loaded, self.tracker.states, self.max_loaded)
            if victim in self.loaded:
                self.loaded.remove(victim)
                evicted.append(victim)
        return {"loaded": list(self.loaded), "evicted": evicted}
