def check(workdir):
    from pte import plan
    import ref

    m = {"parsed": 0.0}
    raw = ref.get_raw_data()
    try:
        out = plan.parse_artifact(raw)
        if isinstance(out, list) and len(out) == 5 and out[0]["id"] == 1:
            m["parsed"] = 1.0
    except Exception:
        pass
    return m
