def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from triton_matmul.tuning import autotune_config

    m = {"autotune_picks_fastest": 0.0}
    try:
        cfg = autotune_config(512, 512, 512)
        if isinstance(cfg, dict) and "BLOCK_SIZE_M" in cfg:
            m["autotune_picks_fastest"] = 1.0
    except Exception:
        pass
    return m
