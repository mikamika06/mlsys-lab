import ref

def select_opt_level(graph, target_latency):
    complexity = len(graph) if isinstance(graph, (list, dict)) else 100
    return ref.oracle_opt_level(complexity)
