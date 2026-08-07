class Node:
    """Dependency graph node representing a layer operator."""

    def __init__(self, name, layer):
        raise NotImplementedError


class Dependency:
    """Directed dependency link between two graph nodes."""

    def __init__(self, src_node, dst_node, src_axis, dst_axis):
        raise NotImplementedError


class DependencyGraph:
    """Graph structure for tracking parameter tensor dependencies."""

    def __init__(self):
        raise NotImplementedError

    def build(self, model):
        raise NotImplementedError
