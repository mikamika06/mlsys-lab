class Node:
    def __init__(self, node_id, op_type, inputs, outputs, attrs=None):
        raise NotImplementedError


class Graph:
    def __init__(self, nodes, inputs, outputs):
        raise NotImplementedError


class Subgraph:
    def __init__(self, sub_id, provider, nodes, inputs, outputs):
        raise NotImplementedError


def partition_graph(graph, supported_ops):
    raise NotImplementedError


def find_partition_breakers(graph, supported_ops):
    raise NotImplementedError
