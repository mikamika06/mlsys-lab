def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import ref
    from fa_fix import dispatcher

    m = {"no_silent_fallback": 0.0}
    configs = []
    for i in range(20):
        dt = "float16" if i % 2 == 0 else "float32"
        al = (i % 3 != 0)
        configs.append({
            "q": ref.create_tensor((1, 4, 64, 32), dtype=dt, aligned=al),
            "k": ref.create_tensor((1, 4, 64, 32), dtype=dt, aligned=al),
            "v": ref.create_tensor((1, 4, 64, 32), dtype=dt, aligned=al)
        })
    try:
        fixed_configs = []
        for cfg in configs:
            q_f, k_f, v_f, m_f = dispatcher.fix_inputs(cfg["q"], cfg["k"], cfg["v"], cfg.get("mask"))
            fixed_configs.append({"q": q_f, "k": k_f, "v": v_f, "mask": m_f})
        res = dispatcher.run_configs(fixed_configs)
        if len(res) == 20 and all(r["backend"] == "flash_attention" for r in res):
            m["no_silent_fallback"] = 1.0
    except Exception:
        pass
    return m
