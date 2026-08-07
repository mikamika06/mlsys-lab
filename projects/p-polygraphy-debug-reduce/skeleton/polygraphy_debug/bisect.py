class GraphBisector:
    def __init__(self, execution_graph, ref_runner, target_runner, tolerance=1e-3):
        raise NotImplementedError

    def bisect(self, inputs):
        raise NotImplementedError
