def check(workdir):
    import ref
    from lora_pipe import engine
    m = {"format_ok": 0.0}
    raw = ref.get_sample_data()
    try:
        res = engine.prepare_data(raw)
        if isinstance(res, list) and len(res) == len(raw) and "tokens" in res[0]:
            m["format_ok"] = 1.0
    except Exception:
        pass
    return m
