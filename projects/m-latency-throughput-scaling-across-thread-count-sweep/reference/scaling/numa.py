"""NUMA topology and access ratio analysis."""


def calculate_numa_ratios(distance_matrix):
    """Calculate relative remote-to-local access latency ratio from a distance matrix."""
    nodes = len(distance_matrix)
    local_sum = 0
    remote_sum = 0
    remote_count = 0

    for i in range(nodes):
        local_sum += distance_matrix[i][i]
        for j in range(nodes):
            if i != j:
                remote_sum += distance_matrix[i][j]
                remote_count += 1

    avg_local = local_sum / nodes
    avg_remote = remote_sum / remote_count if remote_count > 0 else avg_local
    return avg_remote / avg_local


def evaluate_locality_efficiency(access_log, node_distances):
    """Calculate the ratio of local memory access vs total accesses."""
    if not access_log:
        return 1.0
    local_count = sum(1 for thread_node, memory_node in access_log if thread_node == memory_node)
    return local_count / len(access_log)
