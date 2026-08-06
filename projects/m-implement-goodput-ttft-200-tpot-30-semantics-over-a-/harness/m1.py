import ref


def check(workdir):
    from servingmetrics.analysis import compute_goodput
    results = ref.generate_fixtures()
    want_ratio, want_count, want_acc = ref.compute_goodput(results, 200.0, 30.0)
    try:
        got_ratio, got_count, got_acc = compute_goodput(results, 200.0, 30.0)
    except Exception as e:
        return {"goodput_matched": 0.0, "_note": f"raised {type(e).__name__}: {e}"}

    if abs(got_ratio - want_ratio) < 1e-5 and got_count == want_count:
        return {"goodput_matched": 1.0}
    return {"goodput_matched": 0.0, "_note": f"got ratio {got_ratio}, want {want_ratio}"}
