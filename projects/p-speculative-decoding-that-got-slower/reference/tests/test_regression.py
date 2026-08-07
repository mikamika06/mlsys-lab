import sys

sys.path.insert(0, ".")
from specdec.tracker import AcceptanceTracker
from specdec.model import SpeculativeModel
from specdec.policy import AdaptivePolicy


def test_fallback_when_acceptance_drops():
    tr = AcceptanceTracker(window_size=10)
    mod = SpeculativeModel(target_step_cost=10.0, draft_step_cost=1.5, overhead_per_draft=0.2)
    pol = AdaptivePolicy(mod, tr, min_speedup=1.05)

    for _ in range(10):
        tr.record("code", 1, 5)

    gamma, active = pol.decide("code", batch_size=1)
    assert not active or gamma == 0, "Policy should disable speculation when acceptance rate is low"


def test_p95_never_exceeds_baseline_threshold():
    tr = AcceptanceTracker(window_size=20)
    mod = SpeculativeModel(target_step_cost=10.0, draft_step_cost=1.0, overhead_per_draft=0.1)
    pol = AdaptivePolicy(mod, tr, min_speedup=1.02)

    traffic = []
    for i in range(50):
        traffic.append({
            "domain": "chat" if i % 2 == 0 else "code",
            "batch_size": 1 if i < 30 else 32,
            "sim_accepted": 4 if i % 2 == 0 else 0,
            "base_step_time": 10.0
        })

    res = pol.evaluate_p95_and_throughput(traffic)
    assert res["p95_latency"] <= 10.5, f"P95 latency exceeded bound: {res['p95_latency']}"
