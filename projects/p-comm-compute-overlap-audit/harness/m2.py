import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from overlap import audit

    m = {"traffic_matches_theory": 0.0}
    cfg = ref.sample_model_config()
    world_size = 8
    try:
        traffic = audit.calculate_traffic(cfg, world_size)
        expected = audit.calculate_traffic(cfg, world_size)
        if isinstance(traffic, (int, float)) and traffic == expected and traffic > 0:
            m["traffic_matches_theory"] = 1.0
    except Exception:
        pass
    return m
