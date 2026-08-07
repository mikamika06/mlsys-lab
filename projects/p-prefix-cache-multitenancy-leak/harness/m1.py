import ref


def check(workdir):
    m = {"leak_detected": 0.0}
    try:
        c = ref.create_cache(isolate=False)
        tokens = [1, 2, 3, 4]
        c.insert(tokens, tenant_id="tenant_a")
        hits = c.lookup(tokens, tenant_id="tenant_b")
        if hits == len(tokens):
            m["leak_detected"] = 1.0
    except Exception:
        pass
    return m
