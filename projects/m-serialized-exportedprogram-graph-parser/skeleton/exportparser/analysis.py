def eliminate_dead_code(graph_ir):
    """Remove nodes that do not reach any graph output unless marked as side-effecting."""
    raise NotImplementedError


def extract_subgraph(graph_ir, target_nodes):
    """Extract a topological subgraph containing target_nodes and their required inputs."""
    raise NotImplementedError
