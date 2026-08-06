def classify_arguments(arg_names):
    buckets = {"memory": [], "scheduling": [], "latency": [], "correctness": []}
    for name in arg_names:
        if "gpu" in name or "memory" in name or "cache" in name or "block" in name:
            buckets["memory"].append(name)
        elif "sched" in name or "max_num" in name or "batch" in name:
            buckets["scheduling"].append(name)
        elif "tensor" in name or "pipeline" in name or "stream" in name:
            buckets["latency"].append(name)
        else:
            buckets["correctness"].append(name)
    return buckets
