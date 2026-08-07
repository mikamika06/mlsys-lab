import ref


def check(workdir):
    m = {"hit_rate_measured": 0.0}
    try:
        c_global = ref.create_cache(isolate=False)
        c_iso = ref.create_cache(isolate=True)
        tokens = [1, 2, 3, 4, 5]
        c_global.insert(tokens, tenant_id="a")
        c_iso.insert(tokens, tenant_id="a")
        hr_global = c_global.lookup(tokens, tenant_id="b") / len(tokens)
        hr_iso = c_iso.lookup(tokens, tenant_id="b") / len(tokens)
        if hr_global == 1.0 and hr_iso == 0.0:
            m["hit_rate_measured"] = 1.0
    except Exception:
        pass
    return m
