class GraphPass:
    def run(self, graph):
        raise NotImplementedError

class CustomOptimizationPass(GraphPass):
    def run(self, graph):
        new_nodes = []
        for node in graph.get_nodes():
            if node.get("type") == "target_op":
                opt_node = dict(node)
                opt_node["optimized"] = True
                new_nodes.append(opt_node)
            else:
                new_nodes.append(node)
        return type(graph)(new_nodes)

def check_equivalence(graph_orig, graph_opt):
    orig_nodes = graph_orig.get_nodes()
    opt_nodes = graph_opt.get_nodes()
    if len(orig_nodes) != len(opt_nodes):
        return False
    for o, n in zip(orig_nodes, opt_nodes):
        if o.get("id") != n.get("id"):
            return False
    return True

def apply_with_fallback(graph, target_pass):
    try:
        res = target_pass.run(graph)
        if res is None:
            return graph
        return res
    except Exception:
        return graph
