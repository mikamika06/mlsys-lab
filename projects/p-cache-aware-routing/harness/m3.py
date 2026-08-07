import ref


def check(workdir):
    from routing.policy import CacheAwarePolicy
    m = {"policy_ok": 0.0}
    policy = CacheAwarePolicy(2, load_weight=0.5)
    states = [{10, 20}, {10, 20}]
    policy.loads[0] = 5
    policy.loads[1] = 0
    rep = policy.route([10, 20, 30], states)
    if rep == 1:
        m["policy_ok"] = 1.0
    return m
