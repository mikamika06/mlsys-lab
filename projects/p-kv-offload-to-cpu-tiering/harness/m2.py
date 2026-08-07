def check(workdir):
    from kvtier.policy import EvictionPolicy
    m = {"eviction_policy_ok": 0.0}
    p = EvictionPolicy(2)
    sessions = {"a": {"priority": 1, "last_access": 10}, "b": {"priority": 0, "last_access": 5}}
    victim = p.select_victim(sessions)
    if victim == "b":
        m["eviction_policy_ok"] = 1.0
    return m
