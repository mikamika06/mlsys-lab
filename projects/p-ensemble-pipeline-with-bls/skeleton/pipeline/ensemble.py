class PipelineStage:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def run(self, inputs):
        raise NotImplementedError


class EnsembleDAG:
    def __init__(self):
        raise NotImplementedError

    def add_stage(self, name, fn, dependencies=None):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def execute_remote(self, initial_input, network_latency_ms=5.0):
        raise NotImplementedError
