import sys
sys.path.insert(0, ".")
from admission.sim import simulate

def test_time_limit_invariant():
    trace = [{"id": i, "arrival": i // 2, "cost": 10} for i in range(20)]
    max_w = 15
    out = simulate(trace, policy="time_limit", max_wait=max_w)

    for r in out:
        if r["admitted"]:
            assert r["wait"] <= max_w
