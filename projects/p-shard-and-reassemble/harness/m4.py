def check(workdir):
    import ref
    import numpy as np
    try:
        from gguf_shard.sharder import split
        from gguf_shard.engine import Engine
    except ImportError:
        return {"engine_forward_ok": 0.0}

    m = ref.get_test_model_1()
    try:
        shards = split(m, 500)
        engine = Engine(shards)
        x = np.ones((10,), dtype=np.float32)
        y = engine.forward(x, ["l1", "l2", "l3"])

        y_ref = x.copy()
        for name in ["l1", "l2", "l3"]:
            y_ref = y_ref @ m.tensors[name]

        match = 1.0 if np.allclose(y, y_ref) else 0.0
        return {"engine_forward_ok": match}
    except Exception:
        return {"engine_forward_ok": 0.0}
