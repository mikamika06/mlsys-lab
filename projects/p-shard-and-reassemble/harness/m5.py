def check(workdir):
    import ref
    try:
        from gguf_shard.sharder import split
        from gguf_shard.stats import calculate_stats
    except ImportError:
        return {"stats_ok": 0.0}

    m = ref.get_test_model_2()
    try:
        shards = split(m, 1000)
        stats = calculate_stats(shards)

        if stats["total_bytes"] == 125 and stats["total_params"] == 50 and stats["bpw"] == 20.0:
            return {"stats_ok": 1.0}
        return {"stats_ok": 0.0}
    except Exception:
        return {"stats_ok": 0.0}
