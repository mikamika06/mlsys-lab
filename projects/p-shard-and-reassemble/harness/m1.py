def check(workdir):
    import ref
    try:
        from gguf_shard.sharder import split
    except ImportError:
        return {"shard_count_ok": 0.0, "metadata_ok": 0.0}

    m = ref.get_test_model_1()
    try:
        shards = split(m, 500)
        count_ok = 1.0 if len(shards) == 3 else 0.0

        meta_ok = 1.0
        for i, s in enumerate(shards):
            if s.metadata.get("split.no") != i: meta_ok = 0.0
            if s.metadata.get("split.count") != 3: meta_ok = 0.0
            if "split.checksum" not in s.metadata: meta_ok = 0.0
    except Exception:
        count_ok = 0.0
        meta_ok = 0.0

    return {"shard_count_ok": count_ok, "metadata_ok": meta_ok}
