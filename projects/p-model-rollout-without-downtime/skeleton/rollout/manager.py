class ModelManager:
    def __init__(self):
        raise NotImplementedError

    def load_version(self, version_id, model_obj):
        raise NotImplementedError

    def warmup(self, version_id, dummy_inputs):
        raise NotImplementedError

    def switch_traffic(self, version_id, weight):
        raise NotImplementedError

    def rollback(self, fallback_version):
        raise NotImplementedError

    def predict(self, version_id, x):
        raise NotImplementedError
