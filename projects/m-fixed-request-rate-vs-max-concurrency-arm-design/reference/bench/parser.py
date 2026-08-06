import json

def parse_vllm_json(raw_data):
    if isinstance(raw_data, str):
        data = json.loads(raw_data)
    else:
        data = raw_data
    return {
        "duration": float(data.get("duration", 0.0)),
        "completed": int(data.get("completed", 0)),
        "request_throughput": float(data.get("request_throughput", 0.0)),
        "mean_latency_ms": float(data.get("mean_latency_ms", 0.0)),
        "p50_latency_ms": float(data.get("p50_latency_ms", 0.0)),
        "p99_latency_ms": float(data.get("p99_latency_ms", 0.0))
    }
