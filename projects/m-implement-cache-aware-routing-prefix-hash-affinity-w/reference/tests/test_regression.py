import sys
sys.path.insert(0, ".")
from router.capacity import calc_replicas
from router.affinity import select_replica
from router.sim import simulate_trace

def test_capacity_scaling():
    reps_low = calc_replicas(10.0, 5.0, 4)
    reps_high = calc_replicas(100.0, 5.0, 4)
    assert reps_high > reps_low

def test_affinity_guardrail():
    replicas = [
        {"load": 10, "cache": {1}},
        {"load": 2, "cache": set()}
    ]
    chosen = select_replica(replicas, 1, max_load_diff=3)
    assert chosen == 1
