import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from launchbound.profiler import analyze_step
    except ImportError as e:
        return {"step_analysis_matches": 0.0, "_note": f"Import error: {e}"}

    matched = 0
    total = len(ref.WORKLOAD_SPECS)
    for spec in ref.WORKLOAD_SPECS:
        want = ref.analyze_step(**spec)
        try:
            got = analyze_step(**spec)
        except Exception as e:
            return {"step_analysis_matches": 0.0, "_note": f"Execution error: {e}"}

        if len(got) != len(want):
            return {"step_analysis_matches": 0.0, "_note": f"Length mismatch: got {len(got)}, want {len(want)}"}

        ok = True
        for g, w in zip(got, want):
            for k in ["ops", "cpu_time_us", "gpu_time_us", "step_time_us", "gpu_busy_fraction", "is_launch_bound"]:
                if k not in g or abs(g[k] - w[k]) > 1e-5:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            matched += 1

    return {"step_analysis_matches": 1.0 if matched == total else 0.0}
