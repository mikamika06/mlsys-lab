def check(workdir):
    import ref
    from gbreak.analyzer import collect_breaks, group_breaks
    m = {"grouped_correctly": 0.0}
    try:
        breaks = collect_breaks(ref.get_sample_model(), ref.get_sample_inputs())
        grouped = group_breaks(breaks)
        if isinstance(grouped, dict) and len(grouped) > 0:
            m["grouped_correctly"] = 1.0
    except Exception:
        pass
    return m
