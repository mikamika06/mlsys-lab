import ref

def check(workdir):
    m = {"gaps_explained": 0.0}
    try:
        from profiler.analysis import analyze_gaps
        data = ref.get_sample_trace()
        res = analyze_gaps(data)
        if isinstance(res, list) and len(res) > 0:
            m["gaps_explained"] = 1.0
    except Exception:
        pass
    return m
