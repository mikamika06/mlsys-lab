import sys

sys.path.insert(0, ".")
from preempt.recompute import preempt_recompute
from preempt.swap import compute_swap_cost
from preempt.selector import choose_preemption_mode


def test_recompute_preemption_status_and_tokens():
    reqs = [
        {"req_id": "r1", "prompt_len": 100, "generated_len": 20, "num_blocks": 10, "status": "RUNNING"},
        {"req_id": "r2", "prompt_len": 200, "generated_len": 50, "num_blocks": 20, "status": "RUNNING"},
    ]
    updated, tokens = preempt_recompute(reqs, ["r1"])
    assert tokens == 120
    assert updated[0]["status"] == "PREEMPTED"
    assert updated[0]["num_blocks"] == 0
    assert updated[1]["status"] == "RUNNING"
    assert updated[1]["num_blocks"] == 20


def test_swap_cost_roundtrip():
    cost = compute_swap_cost(10, 1024 * 1024, 16.0, roundtrip=True)
    expected_bytes = 20 * 1024 * 1024
    expected_time = expected_bytes / (16.0 * 1e9)
    assert cost["bytes_moved"] == expected_bytes
    assert abs(cost["time_seconds"] - expected_time) < 1e-9


def test_choose_preemption_mode_recompute():
    profile = {
        "num_blocks": 500,
        "block_bytes": 2 * 1024 * 1024,
        "recompute_tokens": 200,
        "token_processing_rate_tps": 20000.0,
        "pcie_bandwidth_gbps": 8.0,
        "roundtrip": True,
    }
    mode = choose_preemption_mode(profile)
    assert mode == "recompute"


def test_choose_preemption_mode_swap():
    profile = {
        "num_blocks": 50,
        "block_bytes": 1024 * 1024,
        "recompute_tokens": 16384,
        "token_processing_rate_tps": 5000.0,
        "pcie_bandwidth_gbps": 64.0,
        "roundtrip": True,
    }
    mode = choose_preemption_mode(profile)
    assert mode == "swap"
