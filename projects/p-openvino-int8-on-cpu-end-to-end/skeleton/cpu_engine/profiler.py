class Profiler:
    def __init__(self, flop_cost_per_ms=300000.0):
        raise NotImplementedError

    def profile(self, graph, input_tensor):
        raise NotImplementedError
