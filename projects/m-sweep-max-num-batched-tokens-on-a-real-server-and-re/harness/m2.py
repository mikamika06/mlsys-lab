import ref


def check(workdir):
    from tokensweep.pareto import find_pareto_front
    workload = ref.generate_workload(123)
    budgets = [256, 512, 1024, 2048, 4096, 8192, 16384]
    results = ref.run_sweep(workload, budgets)
    want = ref.find_pareto_front(results)
    try:
        got = find_pareto_front(results)
    except Exception as e:
        return {"pareto_matched": 0.0, "_note": f"find_pareto_front raised {type(e).__name__}: {e}"}
    if not isinstance(got, list) or len(got) != len(want):
        return {"pareto_matched": 0.0, "_note": f"expected length {len(want)}, got {len(got) if isinstance(got, list) else 'N/A'}"}
    got_budgets = sorted([r["budget"] for r in got])
    want_budgets = sorted([r["budget"] for r in want])
    if got_budgets == want_budgets:
        return {"pareto_matched": 1.0}
    return {"pareto_matched": 0.0, "_note": f"got budgets {got_budgets}, want {want_budgets}"}
