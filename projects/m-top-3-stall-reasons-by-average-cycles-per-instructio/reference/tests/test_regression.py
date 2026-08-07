import sys
sys.path.insert(0, ".")
from warpanalyze.stalls import top_stall_reasons

WARP_STATS = [
    {"reason": "Stall Short Scoreboard", "total_stall_cycles": 10000, "total_executed_instructions": 1000},
    {"reason": "Stall Long Scoreboard", "total_stall_cycles": 5000, "total_executed_instructions": 100},
    {"reason": "Stall Barrier", "total_stall_cycles": 2000, "total_executed_instructions": 200},
    {"reason": "Stall Math Pipe Throttle", "total_stall_cycles": 500, "total_executed_instructions": 100}
]

def test_top_stall_reasons_uses_cpi_not_raw_cycles():
    res = top_stall_reasons(WARP_STATS, k=3)
    reasons = [r["reason"] for r in res]
    assert reasons[0] == "Stall Long Scoreboard", f"Expected Stall Long Scoreboard first, got {reasons[0]}"
    assert reasons[1] == "Stall Short Scoreboard", f"Expected Stall Short Scoreboard second, got {reasons[1]}"
    assert res[0]["avg_cpi"] == 50.0
