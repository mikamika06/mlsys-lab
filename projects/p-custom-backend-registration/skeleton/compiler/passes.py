class GraphPass:
    def run(self, graph):
        raise NotImplementedError

class CustomOptimizationPass(GraphPass):
    def run(self, graph):
        raise NotImplementedError

def check_equivalence(graph_orig, graph_opt):
    raise NotImplementedError

def apply_with_fallback(graph, target_pass):
    raise NotImplementedError
