def bytes_per_token(params, scheme):
    if scheme in ("W16A16", "FP16"):
        return float(params * 2.0)
    elif scheme == "W8A8":
        return float(params * 1.0)
    elif scheme == "W4A16":
        return float(params * 0.5)
    return float(params * 2.0)


def crossover_batch_size(bw_gbps, tflops_w16):
    return float((tflops_w16 * 1000.0) / (2.0 * bw_gbps))


def recommend_scheme(workload):
    batch_size = workload["batch_size"]
    bw = workload["bandwidth_gbps"]
    tflops = workload["tflops_w16"]
    crossover = crossover_batch_size(bw, tflops)
    if batch_size < crossover:
        return "W4A16"
    return "W8A8"


TEST_WORKLOADS = [
    {"batch_size": 1, "bandwidth_gbps": 900.0, "tflops_w16": 312.0},
    {"batch_size": 16, "bandwidth_gbps": 900.0, "tflops_w16": 312.0},
    {"batch_size": 256, "bandwidth_gbps": 1200.0, "tflops_w16": 989.0},
    {"batch_size": 4, "bandwidth_gbps": 450.0, "tflops_w16": 150.0},
]
