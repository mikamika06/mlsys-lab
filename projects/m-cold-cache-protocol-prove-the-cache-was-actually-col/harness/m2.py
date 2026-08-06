import ref


def check(workdir):
    from coldcache.protocol import ColdCacheProtocol
    from coldcache.verifier import verify_cold_execution

    out = {"trace_verified": 0.0}
    proto = ColdCacheProtocol(memory_size=1024)
    reqs = ref.generate_test_requests(seed=123, count=4)

    proof = verify_cold_execution(proto, reqs)

    if proof.get("strictly_cold") is True and proof.get("hits") == 0 and len(proof.get("generations", [])) == 4:
        out["trace_verified"] = 1.0
    else:
        out["_note"] = f"Trace proof failed: {proof}"

    return out
