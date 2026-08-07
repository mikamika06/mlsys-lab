import ref

def check(workdir):
    m = {"report_reproduced": 0.0}
    try:
        from profiler.analysis import generate_phase_report
        data = ref.get_sample_trace()
        res = generate_phase_report(data)
        if isinstance(res, dict) and "inference" in res:
            m["report_reproduced"] = 1.0
    except Exception:
        pass
    return m
