import ref


def check(workdir):
    from kvcache.budget import find_largest_context

    out = {"argmin_index": 0.0}
    budgets = [10000, 50000, 100000, 250000]
    for i, cfg in enumerate(ref.CONFIGS):
        b = budgets[i]
        want = ref.find_largest_context(cfg, b)
        try:
            got = find_largest_context(cfg, b)
        except Exception:
            got = -1
        if got != want:
            out["_note"] = f"config {i}: got {got}, want {want}"
            return out
    out["argmin_index"] = 1.0
    return out
