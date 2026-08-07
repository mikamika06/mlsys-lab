"""Simulate lazy evaluation computation graphs and track memory growth."""

import numpy as np


class LazyGraphNode:

    def __init__(self, shape, parent=None, bytes_per_elem=4):
        self.shape = shape
        self.parent = parent
        self.bytes_per_elem = bytes_per_elem
        self.evaluated = False

    def self_bytes(self):
        return int(np.prod(self.shape)) * self.bytes_per_elem

    def retained_graph_bytes(self):
        total = self.self_bytes()
        curr = self.parent
        while curr is not None and not curr.evaluated:
            total += curr.self_bytes()
            curr = curr.parent
        return total


def simulate_lazy_graph_retention(num_steps, array_size, retain_references):
    base_rss = 20 * 1024 * 1024
    history = []
    retained_nodes = []
    prev_node = None

    for step in range(1, num_steps + 1):
        node = LazyGraphNode(shape=(array_size,), parent=prev_node if retain_references else None)
        if retain_references:
            retained_nodes.append(node)
            prev_node = node
            rss = base_rss + node.retained_graph_bytes()
        else:
            node.evaluated = True
            rss = base_rss + node.self_bytes()

        history.append({
            "step": step,
            "rss_bytes": rss,
            "active_nodes": len(retained_nodes) if retain_references else 1,
        })

    return history


def evaluate_and_clean_graph(node_list):
    freed = 0
    for node in node_list:
        if isinstance(node, LazyGraphNode) and not node.evaluated:
            freed += node.retained_graph_bytes()
            curr = node
            while curr is not None and not curr.evaluated:
                curr.evaluated = True
                curr = curr.parent
    node_list.clear()
    return freed
