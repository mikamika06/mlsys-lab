"""NUMA topology and access ratio analysis."""


def calculate_numa_ratios(distance_matrix):
    """Calculate relative remote-to-local access latency ratio from a distance matrix."""
    raise NotImplementedError


def evaluate_locality_efficiency(access_log, node_distances):
    """Calculate the ratio of local memory access vs total accesses."""
    raise NotImplementedError
