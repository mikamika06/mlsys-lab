def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.selector import generate_recommendation_table
    import ref

    m = {"table_generated": 0.0}
    try:
        configs = [
            {"name": "q4", "bpw": 4.0, "peak_memory_mb": 5000.0},
            {"name": "q8", "bpw": 8.0, "peak_memory_mb": 9000.0}
        ]
        limits = [6.0, 10.0]
        res = generate_recommendation_table(configs, limits)
        ores = ref.oracle_generate_recommendation_table(configs, limits)
        if res == ores:
            m["table_generated"] = 1.0
    except Exception:
        pass
    return m
