from delegate_partition.counter import count_partitions

def compute_latency_curve(graph, base_xnnpack=12.0, base_gpu=6.0):
    p = count_partitions(graph)
    n = len(graph["ops"])
    xnn = [base_xnnpack + (i * 0.25) for i in range(n)]
    gpu = [base_gpu + (p * 1.8) + (i * 0.1) for i in range(n)]
    return {"xnnpack": xnn, "gpu": gpu}
