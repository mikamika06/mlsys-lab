class ModelManager:
    def __init__(self):
        self.versions = {}
        self.active_version = None
        self.loading_version = None
        self.warmup_done = False
        self.traffic_weight = 0.0

    def load_version(self, version_id, model_obj):
        self.loading_version = version_id
        self.versions[version_id] = {"model": model_obj, "loaded": True, "warmed": False}

    def warmup(self, version_id, dummy_inputs):
        if version_id in self.versions:
            for inp in dummy_inputs:
                self.versions[version_id]["model"](inp)
            self.versions[version_id]["warmed"] = True
            self.warmup_done = True

    def switch_traffic(self, version_id, weight):
        if version_id in self.versions and self.versions[version_id]["warmed"]:
            self.traffic_weight = weight
            if weight >= 1.0:
                self.active_version = version_id

    def rollback(self, fallback_version):
        if fallback_version in self.versions:
            self.active_version = fallback_version
            self.traffic_weight = 1.0

    def predict(self, version_id, x):
        if version_id not in self.versions:
            raise ValueError("version not found")
        return self.versions[version_id]["model"](x)
