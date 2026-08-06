import math

def ref_compute_cost_per_million_tokens(throughput_tok_per_sec: float, hourly_instance_price: float) -> float:
    tokens_per_hour = throughput_tok_per_sec * 3600.0
    return (hourly_instance_price / tokens_per_hour) * 1_000_000.0

def ref_compute_required_gpus(hourly_traffic_tok_per_sec: list[float], instance_throughput_tok_per_sec: float, gpus_per_instance: int, headroom_fraction: float = 0.30) -> dict:
    peak_traffic = max(hourly_traffic_tok_per_sec)
    required_capacity = peak_traffic * (1.0 + headroom_fraction)
    instances_needed = math.ceil(required_capacity / instance_throughput_tok_per_sec)
    total_gpus = instances_needed * gpus_per_instance
    return {
        "peak_traffic_tok_per_sec": peak_traffic,
        "target_capacity_tok_per_sec": required_capacity,
        "instances_needed": instances_needed,
        "total_gpus": total_gpus,
    }

def ref_select_cheapest_config(candidates: list[dict], p99_slo_ms: float) -> dict:
    valid = []
    for c in candidates:
        if c["p99_latency_ms"] <= p99_slo_ms:
            cost = ref_compute_cost_per_million_tokens(c["measured_throughput_tok_per_sec"], c["hourly_instance_price"])
            item = dict(c)
            item["cost_per_m_tokens"] = cost
            valid.append(item)
    if not valid:
        raise ValueError("No configuration meets the required p99 SLO")
    valid.sort(key=lambda x: (x["cost_per_m_tokens"], x["p99_latency_ms"]))
    return valid[0]

def generate_test_cases():
    costs_cases = [
        {"throughput": 1500.0, "price": 4.12},
        {"throughput": 850.5, "price": 2.21},
        {"throughput": 3200.0, "price": 12.50},
    ]

    traffic_cases = [
        {
            "curve": [100.0, 250.0, 1200.0, 3500.0, 4200.0, 3900.0, 1500.0, 300.0],
            "throughput": 800.0,
            "gpus": 4,
            "headroom": 0.30
        },
        {
            "curve": [5000.0, 8000.0, 15000.0, 22000.0, 18000.0, 9000.0],
            "throughput": 2500.0,
            "gpus": 8,
            "headroom": 0.25
        }
    ]

    gpu_types = ["A100-80GB", "H100-80GB", "L40S"]
    quants = ["fp16", "fp8"]
    tps = [1, 2]

    candidate_matrix = []
    idx = 0
    for gpu in gpu_types:
        base_price = 3.0 if "A100" in gpu else (5.5 if "H100" in gpu else 1.8)
        for q in quants:
            q_factor = 1.3 if q == "fp8" else 1.0
            for tp in tps:
                idx += 1
                throughput = 600.0 * q_factor * (1.8 if tp == 2 else 1.0) + (idx * 15.0)
                latency = (120.0 / q_factor) * (0.6 if tp == 2 else 1.0) + (idx % 5) * 4.0
                price = base_price * tp
                candidate_matrix.append({
                    "id": f"config_{idx}",
                    "gpu": gpu,
                    "quantization": q,
                    "tensor_parallel": tp,
                    "measured_throughput_tok_per_sec": throughput,
                    "p99_latency_ms": latency,
                    "hourly_instance_price": price
                })

    return costs_cases, traffic_cases, candidate_matrix
