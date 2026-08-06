"""Op-graph diff analysis between ipex.optimize and manual channels_last."""


def diff_op_graphs(manual_graph, ipex_graph):
    """Compare manual channels_last graph vs ipex.optimize graph.

    Returns dict with node_counts, redundant_copies_removed, memory_saved_bytes.
    """
    raise NotImplementedError


def analyze_layout_conversions(graph):
    """Count explicit layout conversion nodes in a graph."""
    raise NotImplementedError
