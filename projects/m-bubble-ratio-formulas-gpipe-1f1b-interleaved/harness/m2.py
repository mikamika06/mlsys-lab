import ref


def check(workdir):
    from pipelib.schedule import Schedule1F1B, ScheduleGPipe

    out = {"schedule_matched": 0.0, "bubble_matched": 0.0}
    matched_sched = 0
    matched_bubble = 0
    total = len(ref.CONFIGS)

    for cfg in ref.CONFIGS:
        p, m = cfg["p"], cfg["m"]
        ref_res = ref.eval_schedules(cfg)

        try:
            gpipe_sim = ScheduleGPipe(p, m, f_cost=1.0, b_cost=2.0).run()
            f1b_sim = Schedule1F1B(p, m, f_cost=1.0, b_cost=2.0).run()
        except Exception as e:
            out["_note"] = f"Schedule execution failed: {e}"
            return out

        if abs(gpipe_sim["makespan"] - ref_res["gpipe_makespan"]) < 1e-5 and \
           abs(f1b_sim["makespan"] - ref_res["f1b_makespan"]) < 1e-5:
            matched_sched += 1

        if abs(gpipe_sim["bubble_ratio"] - ref_res["gpipe_bubble"]) < 1e-5 and \
           abs(f1b_sim["bubble_ratio"] - ref_res["f1b_bubble"]) < 1e-5:
            matched_bubble += 1

    out["schedule_matched"] = 1.0 if matched_sched == total else 0.0
    out["bubble_matched"] = 1.0 if matched_bubble == total else 0.0
    return out
