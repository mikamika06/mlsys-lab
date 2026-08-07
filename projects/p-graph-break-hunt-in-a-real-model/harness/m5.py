def check(workdir):
    import ref
    from gbreak.optimizer import count_graphs
    m = {"graph_count_ok": 0.0}
    try:
        cnt = count_graphs(ref.get_sample_model(), ref.get_sample_inputs())
        if isinstance(cnt, (int, float)) and cnt <= 5:
            m["graph_count_ok"] = 1.0
    except Exception:
        pass
    return m
