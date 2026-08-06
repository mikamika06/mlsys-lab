import ref


def check(workdir):
    from chash.router import ConsistentHashRing
    from chash.diagnostics import diagnose_hot_replicas
    from chash.affinity import SessionAffinityRouter

    keys, logs = ref.generate_routing_dataset()
    diag = diagnose_hot_replicas(logs, threshold_std_dev=1.5)

    hot_ok = 1.0 if ("r3" in diag.get("hot_replicas", [])) else 0.0

    ring = ConsistentHashRing(["r1", "r2", "r3", "r4"], num_tokens=100)
    router = SessionAffinityRouter(ring, ttl_seconds=60)

    sessions = {
        "s1": keys[:10],
        "s2": keys[10:20],
        "s3": keys[20:30]
    }

    res = router.evaluate_affinity_ttl(sessions, total_duration=50, key_churn_rate=0.2)
    optimal_ttl = res.get("optimal_ttl", 0)

    affinity_ok = 1.0 if (5 <= optimal_ttl <= 120) else 0.0

    return {
        "hot_replica_detected": float(hot_ok),
        "affinity_ttl_optimal": float(affinity_ok)
    }
