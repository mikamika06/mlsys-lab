import json

SAMPLE_JSONS = [
    '{"duration": 10.0, "completed": 100, "request_throughput": 10.0, "mean_latency_ms": 50.0, "p50_latency_ms": 45.0, "p99_latency_ms": 120.0}',
    '{"duration": 15.0, "completed": 150, "request_throughput": 10.0, "mean_latency_ms": 55.0, "p50_latency_ms": 50.0, "p99_latency_ms": 130.0}',
    '{"duration": 20.0, "completed": 200, "request_throughput": 10.0, "mean_latency_ms": 48.0, "p50_latency_ms": 42.0, "p99_latency_ms": 115.0}'
]

SAMPLE_RUNS_FR = [
    {"request_throughput": 9.5, "p99_latency_ms": 110.0},
    {"request_throughput": 9.8, "p99_latency_ms": 112.0}
]

SAMPLE_RUNS_MC = [
    {"request_throughput": 14.2, "p99_latency_ms": 165.0},
    {"request_throughput": 14.5, "p99_latency_ms": 170.0}
]

def parse_vllm_json(raw_data):
    d = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    return {
        "duration": float(d.get("duration", 0.0)),
        "completed": int(d.get("completed", 0)),
        "request_throughput": float(d.get("request_throughput", 0.0)),
        "mean_latency_ms": float(d.get("mean_latency_ms", 0.0)),
        "p50_latency_ms": float(d.get("p50_latency_ms", 0.0)),
        "p99_latency_ms": float(d.get("p99_latency_ms", 0.0))
    }
