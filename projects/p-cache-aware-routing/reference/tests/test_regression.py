import sys
sys.path.insert(0, ".")
from routing.router import Router
from routing.policy import CacheAwarePolicy


def test_round_robin_hit_rate_low():
    r = Router(3)
    prompts = [[1, 2, 3], [1, 2, 4], [1, 2, 5]]
    hits = 0
    for i, p in enumerate(prompts):
        rep = r.round_robin_route(i)
        if len(r.replica_states[rep].intersection(p)) > 0:
            hits += 1
        r.replica_states[rep].update(p)
    assert hits == 0


def test_affinity_computation():
    from routing.affinity import compute_affinity
    state = {1, 2, 3}
    assert compute_affinity([1, 2, 4], state) == 2 / 3


def test_policy_load_balancing():
    pol = CacheAwarePolicy(2, load_weight=0.1)
    states = [set(), set()]
    r0 = pol.route([1, 2], states)
    r1 = pol.route([1, 2], states)
    assert r0 != r1 or pol.loads[0] != pol.loads[1]


def test_replica_failure_handling():
    pol = CacheAwarePolicy(2)
    pol.loads[0] = 10
    pol.loads[1] = 0
    rep = pol.route([1, 2], [set([1, 2]), set([1, 2])])
    assert rep == 1
