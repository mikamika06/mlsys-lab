import ref


def check(workdir):
    from memplan.config import evaluate_feasibility, optimization_schedule
    from memplan.evaluator import calculate_overhead, verify_memory_bounds

    out = {
        "bytes_match": 0.0,
        "feasibility_match": 0.0,
        "saves_memory": 0.0,
        "schedule_match": 0.0
    }

    cfg = ref.CONFIGS[0]
    ram = 1000000000

    got_feas = evaluate_feasibility(cfg, ram)
    want_feas = ref.evaluate_feasibility(cfg, ram)
    if got_feas == want_feas:
        out["feasibility_match"] = 1.0

    got_sched = optimization_schedule(512, 4)
    want_sched = ref.optimization_schedule(512, 4)
    if got_sched == want_sched:
        out["schedule_match"] = 1.0

    ov = calculate_overhead(512, 1024, 256)
    if isinstance(ov, (int, float)) and ov > 0:
        out["bytes_match"] = 1.0

    bounds_ok = verify_memory_bounds(5000, 10000)
    if bounds_ok is True:
        out["saves_memory"] = 1.0

    return out
