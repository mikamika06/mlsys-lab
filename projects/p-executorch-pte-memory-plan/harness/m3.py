def check(workdir):
    from pte import plan
    import ref

    m = {"separated": 0.0}
    raw = ref.get_raw_data()
    try:
        tensors = plan.parse_artifact(raw)
        c, a = plan.split_program_data(tensors)
        if len(c) == 1 and c[0]["constant"] and len(a) == 4 and not a[0]["constant"]:
            m["separated"] = 1.0
    except Exception:
        pass
    return m
