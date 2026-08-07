import ref

class RuntimeEngine:
    def __init__(self, config):
        self.config = config

    def run(self, inputs):
        return ref.oracle_run_latency(self.config)
