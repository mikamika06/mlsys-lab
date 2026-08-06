import sys
sys.path.insert(0, ".")
from hpa.sim import simulate_hpa, diagnose_thrash
from hpa.affinity import evaluate_session_affinity


def test_simulation_bounds():
    trace = [100, 200, 300, 200, 100]
    res = simulate_hpa(trace, target_util=80, stabilization_window=2, min_replicas=1, max_replicas=10)
    assert len(res) == len(trace)
    assert all(1 <= r <= 10 for r in res)


def test_affinity_loss():
    sessions = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    hit_rate_rand = evaluate_session_affinity(sessions, num_replicas=4, routing_strategy="random")
    assert 0.0 <= hit_rate_rand <= 1.0
