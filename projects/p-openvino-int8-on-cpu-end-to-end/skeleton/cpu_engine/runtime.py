class RuntimeEngine:
    def __init__(self, graph):
        raise NotImplementedError

    def configure(self, num_threads=1, hint="NONE", enable_numa=False):
        raise NotImplementedError

    def infer(self, input_tensor):
        raise NotImplementedError

    def generate_waterfall_report(self, input_tensor):
        raise NotImplementedError
