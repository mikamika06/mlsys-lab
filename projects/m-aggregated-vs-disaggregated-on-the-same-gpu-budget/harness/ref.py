import random
from disagg.simulator import Request, simulate_aggregated, simulate_disaggregated
from disagg.metrics import compute_latency_ratios


def generate_workload(seed: int = 42, num_requests: int = 50) -> list[Request]:
    rng = random.Random(seed)
    reqs = []
    curr_time = 0.0
    for i in range(num_requests):
        curr_time += rng.expovariate(100.0)
        prompt_len = rng.randint(128, 2048)
        decode_len = rng.randint(16, 128)
        reqs.append(Request(req_id=i, arrival_time=curr_time, prompt_len=prompt_len, decode_len=decode_len))
    return reqs


def get_reference_results():
    reqs = generate_workload(42, 50)
    num_gpus = 8
    p_gpus = 2
    d_gpus = 6
    prefill_rate = 5000.0
    decode_rate = 200.0
    kv_rate = 10e7

    agg = simulate_aggregated(reqs, num_gpus, prefill_rate, decode_rate)
    disagg = simulate_disaggregated(reqs, p_gpus, d_gpus, prefill_rate, decode_rate, kv_rate)
    metrics = compute_latency_ratios(agg, disagg)
    return agg, disagg, metrics
