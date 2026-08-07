def plan_residency(model_a, model_b, wired_limit_mb):
    limit_bytes = wired_limit_mb * 1024 * 1024
    size_a = model_a.get("weight_bytes", 0) + model_a.get("kv_bytes", 0)
    size_b = model_b.get("weight_bytes", 0) + model_b.get("kv_bytes", 0)

    if size_a + size_b <= limit_bytes:
        return [{"step": i, "model_a_resident": True, "model_b_resident": True, "wired_bytes": size_a + size_b} for i in range(3)]

    schedule = []
    for i in range(4):
        if i % 2 == 0:
            res_a = True
            res_b = (size_b <= limit_bytes)
            w = size_a if res_b is False else min(limit_bytes, size_a + size_b)
        else:
            res_a = (size_a <= limit_bytes)
            res_b = True
            w = size_b if res_a is False else min(limit_bytes, size_a + size_b)
        schedule.append({"step": i, "model_a_resident": res_a, "model_b_resident": res_b, "wired_bytes": w})
    return schedule
