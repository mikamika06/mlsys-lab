def check(workdir):
    import ref
    try:
        from gguf_shard.sharder import split, verify_shard
    except ImportError:
        return {"verifies_good": 0.0, "rejects_bad": 0.0}

    m = ref.get_test_model_1()
    try:
        shards = split(m, 500)
        good = verify_shard(shards[0])

        first_key = list(shards[0].tensors.keys())[0]
        shards[0].tensors[first_key][0, 0] += 999.0
        bad = verify_shard(shards[0])

        return {
            "verifies_good": 1.0 if good else 0.0,
            "rejects_bad": 1.0 if not bad else 0.0
        }
    except Exception:
        return {"verifies_good": 0.0, "rejects_bad": 0.0}
