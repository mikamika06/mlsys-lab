class GraphTracer:
    def __init__(self):
        raise NotImplementedError

    def trace_graph(self, ops):
        raise NotImplementedError

    def inspect_fusions(self, graph, expected_fusions):
        raise NotImplementedError

    def find_unfused_nodes(self, graph):
        raise NotImplementedError
