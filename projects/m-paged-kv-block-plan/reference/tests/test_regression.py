import sys
sys.path.insert(0, ".")

from kvplan.planner import calculate_paged_kv_plan, simulate_block_allocation
from kvplan.bench import generate_throughput_report


def test_paged_kv_plan_math():
    plan = calculate_paged_kv_plan([128, 256], 16, 100)
    assert plan["total_blocks_needed"] == 8 + 16
    assert plan["allocated_blocks"] == 24
    assert plan["fits_in_budget"] is True


def test_block_allocation_simulation():
    pattern = [
        {"action": "arrive", "seq_len": 64},
        {"action": "arrive", "seq_len": 64},
        {"action": "depart", "seq_len": 64}
    ]
    sim = simulate_block_allocation(pattern, 16, 10)
    assert sim["completed"] == 2
    assert sim["active_blocks"] == 4


def test_throughput_report_validity():
    rep = generate_throughput_report(10, 1000, 1000, 10.0)
    assert rep["token_throughput"] == 200.0
    assert rep["throughput_ratio"] == 2.0
