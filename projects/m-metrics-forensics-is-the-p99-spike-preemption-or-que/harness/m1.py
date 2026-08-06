import ref


def check(workdir):
    from forensics.metrics import analyze_metrics as learner_analyze

    cases = ref.generate_trace_cases()
    matched = 0
    for i, case in enumerate(cases):
        ref_out = ref.analyze_metrics(case)
        try:
            got_out = learner_analyze(case)
        except Exception as e:
            return {"metrics_matched": float(matched), "_note": f"case {i} raised {e}"}

        if got_out is None:
            continue

        if (abs(got_out.get("mean_queue_wait", -1) - ref_out["mean_queue_wait"]) < 1e-3 and
            abs(got_out.get("p99_queue_wait", -1) - ref_out["p99_queue_wait"]) < 1e-3 and
            got_out.get("preemption_events") == ref_out["preemption_events"] and
            got_out.get("is_preemption_dominant") == ref_out["is_preemption_dominant"]):
            matched += 1

    return {"metrics_matched": float(matched)}
