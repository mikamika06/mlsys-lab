import ref


def check(workdir):
    m = {"recovery_ok": 0.0}
    try:
        c = ref.create_cache(isolate=True)
        sys_pfx = [1, 2]
        tokens = [1, 2, 3, 4, 5]
        c.insert(tokens, tenant_id="a", system_prefixes=[sys_pfx])
        hits = c.lookup(tokens, tenant_id="b", system_prefixes=[sys_pfx])
        if hits >= 2:
            m["recovery_ok"] = 1.0
    except Exception:
        pass
    return m
