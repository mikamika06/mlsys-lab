import random

def generate_mock_exposition():
    random.seed(42)
    lines = [
        "# HELP vllm:request_latency_seconds Request latency",
        "# TYPE vllm:request_latency_seconds histogram"
    ]
    cumulative = 0
    les = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, "+Inf"]
    counts = [12, 45, 120, 350, 780, 920, 980, 995, 1000]
    for le, count in zip(les, counts):
        lines.append(f'vllm:request_latency_seconds_bucket{{le="{le}",model="llama3"}} {count}.0')
    lines.append('vllm:request_latency_seconds_sum 2450.5')
    lines.append('vllm:request_latency_seconds_count 1000.0')
    return "\n".join(lines)

def generate_client_latencies():
    random.seed(123)
    return [random.expovariate(1.5) for _ in range(500)]
