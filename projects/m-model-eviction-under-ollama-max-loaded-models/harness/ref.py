class ReferenceTracker:
    def __init__(self):
        self.states = {}

    def touch(self, model_name, time_step):
        self.states[model_name] = time_step


class ReferencePolicy:
    @staticmethod
    def select_evict(loaded_models, access_times, max_loaded):
        if len(loaded_models) <= max_loaded:
            return None
        sorted_models = sorted(loaded_models, key=lambda m: access_times.get(m, 0))
        return sorted_models[0]


class ReferenceManager:
    def __init__(self, max_loaded):
        self.max_loaded = max_loaded
        self.loaded = []
        self.access_times = {}
        self.time = 0

    def request(self, model_name):
        self.time += 1
        self.access_times[model_name] = self.time
        if model_name not in self.loaded:
            self.loaded.append(model_name)
        evicted = []
        while len(self.loaded) > self.max_loaded:
            victim = ReferencePolicy.select_evict(self.loaded, self.access_times, self.max_loaded)
            if victim in self.loaded:
                self.loaded.remove(victim)
                evicted.append(victim)
        return {"loaded": list(self.loaded), "evicted": evicted}


def get_test_scenarios():
    return [
        {"max_loaded": 2, "requests": ["llama3", "mistral", "gemma", "llama3"]},
        {"max_loaded": 1, "requests": ["phi3", "llama3", "phi3"]},
        {"max_loaded": 3, "requests": ["m1", "m2", "m3", "m4", "m1", "m5"]}
    ]


def run_reference(scenario):
    mgr = ReferenceManager(scenario["max_loaded"])
    history = []
    for req in scenario["requests"]:
        res = mgr.request(req)
        history.append(res)
    return history
