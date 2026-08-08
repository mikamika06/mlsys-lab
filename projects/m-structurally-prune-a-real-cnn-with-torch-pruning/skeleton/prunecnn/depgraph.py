def build_depgraph(config):
    """Build dependency relations between layer inputs and outputs."""
    raise NotImplementedError


def get_pruning_group(config, trigger_layer, prune_channels):
    """Propagate channel pruning trigger through the dependency graph."""
    raise NotImplementedError
