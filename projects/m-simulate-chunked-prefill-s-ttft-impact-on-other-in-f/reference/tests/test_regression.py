import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_sched.scheduler import simulate_schedule

def test_chunked_prefill_limits_stall():
    res = simulate_schedule(
        prompt_len=1000,
        inflight_reqs=[10, 10],
        chunk_size=200,
        prefill_cost=0.1,
        decode_cost=1.0
    )
    assert res["max_stall"] <= 20.0, f"Stalled for {res['max_stall']} ms"

def test_ttft_computation():
    res = simulate_schedule(300, [10, 10], 200, 0.1, 1.0)
    assert abs(res["ttft"] - 32.0) < 1e-5, f"TTFT was {res['ttft']}"
