import ref


def check(workdir):
    from chash.router import ConsistentHashRing, calculate_remapping_fraction

    keys, _ = ref.generate_routing_dataset()
    oracle_4 = ref.OracleConsistentHashRing(["r1", "r2", "r3", "r4"], num_tokens=100)
    oracle_5 = ref.OracleConsistentHashRing(["r1", "r2", "r3", "r4", "r5"], num_tokens=100)

    expected_remap = 0.0
    for k in keys:
        if oracle_4.get_replica(k) != oracle_5.get_replica(k):
            expected_remap += 1.0
    expected_remap /= len(keys)

    learner_4 = ConsistentHashRing(["r1", "r2", "r3", "r4"], num_tokens=100)
    learner_5 = ConsistentHashRing(["r1", "r2", "r3", "r4", "r5"], num_tokens=100)

    got_remap = calculate_remapping_fraction(learner_4, learner_5, keys)
    rel_err = abs(got_remap - expected_remap) / (expected_remap + 1e-9)

    remapping_ok = 1.0 if (0.15 <= got_remap <= 0.30) else 0.0

    return {
        "remapping_ratio_ok": float(remapping_ok),
        "rel_err": float(rel_err)
    }
