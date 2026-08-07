import ref


def check(workdir):
    from vllm_diag.isolation import check_cache_salt_isolation
    tokens = [1, 2, 3, 4]
    want = ref.verify_isolation("tenant_a", "tenant_b", tokens)
    got = check_cache_salt_isolation("tenant_a", "tenant_b", tokens)
    out = {"isolation_verified": 1.0 if got == want else 0.0}
    if got != want:
        out["_note"] = f"isolation check failed for salts tenant_a and tenant_b"
    return out
