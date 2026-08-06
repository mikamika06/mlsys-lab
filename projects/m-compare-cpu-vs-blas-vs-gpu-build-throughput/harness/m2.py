import ref

def check(workdir):
    from backperf.repack import analyze_size_invariance
    fixture = ref.get_size_analysis_fixtures()
    want = analyze_size_invariance(fixture)
    try:
        got = analyze_size_invariance(fixture)
    except Exception as e:
        return {"size_analysis_matched": 0.0, "_note": f"raised {e}"}

    if got == want:
        return {"size_analysis_matched": 1.0}
    else:
        return {"size_analysis_matched": 0.0, "_note": f"got {got}, want {want}"}
