def check(workdir):
    import ref
    try:
        from gguf_shard.sharder import split, reassemble
        import numpy as np
    except ImportError:
        return {"reassembled_matches": 0.0}

    m = ref.get_test_model_1()
    try:
        shards = split(m, 500)
        m2 = reassemble(shards)

        match = 1.0
        if "split.no" in m2.metadata: match = 0.0
        for k in m.tensors:
            if not np.allclose(m.tensors[k], m2.tensors[k]):
                match = 0.0
        return {"reassembled_matches": match}
    except Exception:
        return {"reassembled_matches": 0.0}
