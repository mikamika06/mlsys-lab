import ref


def check(workdir):
    m = {"zero_intersections": 0.0}
    try:
        c = ref.create_cache(isolate=True)
        tokens_a = [10, 20, 30, 40]
        tokens_b = [10, 20, 99, 88]
        c.insert(tokens_a, tenant_id="tenant_a")
        hits = c.lookup(tokens_b, tenant_id="tenant_b")
        if hits == 0:
            m["zero_intersections"] = 1.0
    except Exception:
        pass
    return m
