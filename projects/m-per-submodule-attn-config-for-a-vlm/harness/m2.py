import ref


def check(workdir):
    from vlmcfg.memory import plan_bytes, uniform_bytes, free_schedule

    out = {"bytes_match": 0.0, "uniform_match": 0.0, "saves_memory": 0.0, "schedule_match": 0.0}
    cfg = ref.CONFIGS[2]

    got_bytes = plan_bytes(cfg, 4096, 2, 1)
    want_bytes = ref.plan_bytes(cfg, 4096, 2, 1)
    if got_bytes == want_bytes:
        out["bytes_match"] = 1.0

    got_uniform = uniform_bytes(cfg, 4096, 2, 1)
    want_uniform = ref.uniform_bytes(cfg, 4096, 2, 1)
    if got_uniform == want_uniform:
        out["uniform_match"] = 1.0

    if got_bytes < got_uniform:
        out["saves_memory"] = 1.0

    got_sched = free_schedule(4096, 2, 10)
    want_sched = ref.free_schedule(4096, 2, 10)
    if got_sched == want_sched:
        out["schedule_match"] = 1.0

    return out
