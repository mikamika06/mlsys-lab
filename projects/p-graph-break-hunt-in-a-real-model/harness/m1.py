def check(workdir):
    import ref
    from gbreak.analyzer import collect_breaks
    m = {"collected_all": 0.0}
    try:
        breaks = collect_breaks(ref.get_sample_model(), ref.get_sample_inputs())
        if isinstance(breaks, list) and len(breaks) >= 4:
            m["collected_all"] = 1.0
    except Exception:
        pass
    return m
