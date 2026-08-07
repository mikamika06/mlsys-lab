from preemption.simulator import simulate_wasted_tokens
from preemption.model import find_breakeven_point


def get_m1_data():
    reqs = [
        {"id": 10, "prefill_len": 256},
        {"id": 20, "prefill_len": 512},
        {"id": 30, "prefill_len": 1024}
    ]
    preempts = [
        {"request_id": 10, "step": 12},
        {"request_id": 30, "step": 15},
        {"request_id": 10, "step": 20}
    ]
    expected = simulate_wasted_tokens(reqs, preempts)
    return reqs, preempts, expected


def get_m2_data():
    kv_bytes = 2048
    bw = 40.0
    cost = 0.1
    expected = find_breakeven_point(kv_bytes, bw, cost)
    return kv_bytes, bw, cost, expected
