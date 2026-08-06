import ref


def check(workdir):
    out = {"levels_matched": 0.0}
    try:
        from ortopt.levels import select_cheapest_level
    except Exception as e:
        out["_note"] = f"Import error: {type(e).__name__}: {e}"
        return out

    dataset = ref.generate_level_data()
    passed = 0
    total = len(dataset)

    for latencies, setup_costs, tol in dataset:
        want = ref.ref_select_cheapest_level(latencies, setup_costs, tol)
        try:
            got = select_cheapest_level(latencies, setup_costs, tol)
        except Exception as e:
            out["_note"] = f"Execution error: {type(e).__name__}: {e}"
            return out

        if got == want:
            passed += 1
        elif "_note" not in out:
            out["_note"] = f"Mismatch: expected level {want}, got {got} for latencies {latencies}"

    if passed == total:
        out["levels_matched"] = 1.0
    return out
