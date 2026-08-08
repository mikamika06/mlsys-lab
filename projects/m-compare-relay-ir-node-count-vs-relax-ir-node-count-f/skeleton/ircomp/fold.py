def fold_relay_constants(ast_dict):
    """Simulate constant folding pass on Relay AST and return folded AST and node count."""
    raise NotImplementedError


def fold_relax_constants(ast_dict):
    """Simulate constant folding pass on Relax AST and return folded AST and node count."""
    raise NotImplementedError


def analyze_folding_divergence(subgraphs):
    """Compute node reduction counts and divergence ratios between Relay and Relax."""
    raise NotImplementedError
