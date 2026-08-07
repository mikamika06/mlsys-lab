import ref


def check(workdir):
    m = {"isolated_hash_ok": 0.0}
    try:
        c = ref.create_cache(isolate=True)
        tokens = [1, 2, 3, 4]
        c.insert(tokens, tenant_id="tenant_a")
        hits_a = c.lookup(tokens, tenant_id="tenant_a")
        hits_b = c.lookup(tokens, tenant_id="tenant_b")
        if hits_a == len(tokens) and hits_b == 0:
            m["isolated_hash_ok"] = 1.0
    except Exception:
        pass
    return m
