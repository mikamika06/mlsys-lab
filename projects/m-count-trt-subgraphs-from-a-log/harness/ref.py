LOGS = [
    "I [VFX] [TensorRTExecutionProvider] Ingress node parsing start.\n"
    "I [VFX] [TensorRTExecutionProvider] Subgraph 0: 12 nodes partitioned.\n"
    "I [VFX] [TensorRTExecutionProvider] Subgraph 1: 5 nodes partitioned.\n"
    "I [VFX] [TensorRTExecutionProvider] Build engine successfully, timing cache loaded from disk.\n",
    "I [VFX] [TensorRTExecutionProvider] Subgraph 0: 30 nodes partitioned.\n"
    "I [VFX] [TensorRTExecutionProvider] No timing cache found, building from scratch.\n",
    "I [VFX] [TensorRTExecutionProvider] Subgraph 0: 8 nodes partitioned.\n"
    "I [VFX] [TensorRTExecutionProvider] Subgraph 1: 4 nodes partitioned.\n"
    "I [VFX] [TensorRTExecutionProvider] Subgraph 2: 15 nodes partitioned.\n"
    "I [VFX] [TensorRTExecutionProvider] Timing cache size: 1024 bytes, reused.\n"
]

def parse_log(text):
    subgraphs = []
    cache_reused = False
    for line in text.splitlines():
        if "Subgraph" in line:
            parts = line.split(":")
            sub_id = parts[0].strip()
            nodes = int(parts[1].strip().split()[0])
            subgraphs.append({"id": sub_id, "nodes": nodes})
        if "reused" in line.lower() or "loaded" in line.lower():
            cache_reused = True
    return {"subgraphs": subgraphs, "count": len(subgraphs), "cache_reused": cache_reused}

def compare_eps(cuda_latencies, trt_latencies):
    c_mean = sum(cuda_latencies) / len(cuda_latencies)
    t_mean = sum(trt_latencies) / len(trt_latencies)
    return {"cuda_mean": c_mean, "trt_mean": t_mean, "latency_ratio": c_mean / t_mean}

def simulate_cache_sessions(session_logs):
    results = []
    for log in session_logs:
        parsed = parse_log(log)
        results.append(parsed["cache_reused"])
    return results
