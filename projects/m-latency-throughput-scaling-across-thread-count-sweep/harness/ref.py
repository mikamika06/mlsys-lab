"""Reference data generation and verification oracle."""

TOPOLOGY = {
    "p_cores": 8,
    "e_cores": 2,
    "total_cores": 10
}

SWEEP_LATENCIES = {
    1: 100.0,
    2: 52.0,
    4: 28.0,
    8: 16.0,
    9: 22.0,
    10: 30.0,
    12: 48.0,
    16: 75.0
}

WORK_UNITS = 1000

DISTANCE_MATRIX = [
    [10, 21, 31, 31],
    [21, 10, 31, 31],
    [31, 31, 10, 21],
    [31, 31, 21, 10]
]

ACCESS_LOG = [
    (0, 0), (0, 0), (0, 1), (0, 0),
    (1, 1), (1, 1), (1, 0), (1, 1),
    (2, 2), (2, 3), (2, 2), (2, 2),
    (3, 3), (3, 3), (3, 2), (3, 3)
]


def find_oversubscription_point(topology, latency_data):
    p_cores = topology.get("p_cores", 0)
    for threads in sorted(latency_data.keys()):
        if threads > p_cores:
            return threads
    return p_cores + 1


def analyze_thread_sweep(latency_data, work_units):
    base_latency = latency_data[1]
    result = {}
    for threads, latency_ms in sorted(latency_data.items()):
        throughput = work_units / (latency_ms / 1000.0)
        latency_ratio = latency_ms / base_latency
        result[threads] = {
            "throughput": throughput,
            "latency_ratio": latency_ratio
        }
    return result


def calculate_numa_ratios(distance_matrix):
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
    if not access_log:
        return 1.0
    local_count = sum(1 for thread_node, memory_node in access_log if thread_node == memory_node)
    return local_count / len(access_log)
