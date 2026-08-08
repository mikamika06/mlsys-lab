def count_relay_nodes(ast_dict):
    """Count total AST nodes in a Relay IR AST dictionary representation."""
    raise NotImplementedError


def count_relax_nodes(ast_dict):
    """Count total AST nodes in a Relax IR AST dictionary representation."""
    raise NotImplementedError


def compare_node_counts(subgraphs):
    """Return a list of dicts comparing Relay vs Relax node counts for each subgraph."""
    raise NotImplementedError
