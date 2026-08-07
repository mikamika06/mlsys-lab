class BakeoffRunner:
    def __init__(self, config):
        raise NotImplementedError

    def run_engine(self, engine_name):
        raise NotImplementedError

    def get_prefill_metrics(self, engine_name):
        raise NotImplementedError

    def get_decode_metrics(self, engine_name):
        raise NotImplementedError

    def get_resource_usage(self, engine_name):
        raise NotImplementedError

    def evaluate_stability(self, runs=3):
        raise NotImplementedError

    def recommend(self):
        raise NotImplementedError
