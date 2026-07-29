import ref


def check(workdir):
    from kvplan import free_schedule, plan_bytes, uniform_bytes

    out = {"bytes_match": 0.0, "uniform_match": 0.0, "saves_memory": 0.0, "schedule_match": 0.0}
    bad = []
    for i, cfg in enumerate(ref.CONFIGS):
        for ctx, bs, elem in ((8192, 16, 2), (1024, 32, 1)):
            w = ref.plan_bytes(cfg, ctx, bs, elem)
            g = plan_bytes(cfg, ctx, bs, elem)
            if g != w:
                bad.append(f"config {i} ctx {ctx}: {g} != {w}")
    out["bytes_match"] = 1.0 if not bad else 0.0
    out["uniform_match"] = 1.0 if all(
        uniform_bytes(c, 8192, 16, 2) == ref.uniform_bytes(c, 8192, 16, 2)
        for c in ref.CONFIGS) else 0.0
    c0 = ref.CONFIGS[0]
    out["saves_memory"] = 1.0 if plan_bytes(c0, 8192, 16, 2) < uniform_bytes(c0, 8192, 16, 2) else 0.0
    out["schedule_match"] = 1.0 if all(
        list(free_schedule(w, b, n)) == ref.free_schedule(w, b, n)
        for w, b, n in ((512, 16, 2000), (256, 32, 900), (4096, 16, 100))) else 0.0
    if bad:
        out["_note"] = "; ".join(bad[:2])
    return out
