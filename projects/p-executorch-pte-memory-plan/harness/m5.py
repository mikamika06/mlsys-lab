def check(workdir):
    from pte import plan
    import ref

    m = {"budget_checked": 0.0}
    raw = ref.get_large_data()
    try:
        tensors = plan.parse_artifact(raw)
        ok1 = plan.check_budget(tensors, 600)
        ok2 = plan.check_budget(tensors, 599)
        if ok1 and not ok2:
            m["budget_checked"] = 1.0
    except Exception:
        pass
    return m
