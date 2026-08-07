import numpy as np

class GroupFinder:
    def __init__(self, graph):
        self.graph = graph

    def find_groups(self):
        parent = {node: node for node in self.graph.nodes}
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for src, dst, _, _ in self.graph.edges:
            union(src, dst)

        groups = {}
        for node in self.graph.nodes:
            root = find(node)
            if root not in groups:
                groups[root] = []
            groups[root].append(node)
        return list(groups.values())
