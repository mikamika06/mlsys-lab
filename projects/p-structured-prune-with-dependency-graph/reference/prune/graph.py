import numpy as np

class DependencyGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, name, shape):
        self.nodes[name] = shape

    def add_edge(self, src, dst, dim_src, dim_dst):
        self.edges.append((src, dst, dim_src, dim_dst))
