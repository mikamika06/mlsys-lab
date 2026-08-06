import ref


def check(workdir):
    from servingmetrics.analysis import analyze_preemption_gap
    results = ref.generate_fixtures()
    sample = results[0]
    want = ref.analyze_preemption_gap(sample)
    try:
        got = analyze_preemption_gap(sample)
    except Exception as e:
        return {"gap_matched": 0.0, "_note": f"raised {type(e).__name__}: {e}"}

    if got and abs(got.get("actual_gap", 0) - want.get("actual_gap", 0)) < 1e-5:
        return {"gap_matched": 1.0}
    return {"gap_matched": 0.0, "_note": f"got {got}, want {want}"}
