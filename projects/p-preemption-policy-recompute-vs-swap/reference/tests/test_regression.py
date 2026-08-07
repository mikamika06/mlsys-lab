import sys
sys.path.insert(0, ".")
from preempt.model import recompute_cost, swap_cost, breakeven_length
from preempt.policy import PreemptionPolicy
from preempt.scheduler import PreemptionScheduler

def test_recompute_cost_positive():
    c = recompute_cost(1024, 4096, 32, 300.0)
    assert c > 0

def test_swap_cost_positive():
    c = swap_cost(1024, 131072, 32.0)
    assert c > 0

def test_breakeven_finite():
    be = breakeven_length(4096, 32, 300.0, 131072, 32.0)
    assert be > 0

def test_scheduler_runs():
    sch = PreemptionScheduler({"max_running": 2})
    sch.run_trace({0: [{"prompt": [1, 2], "total_len": 10}]})
    assert len(sch.latencies) == 1
