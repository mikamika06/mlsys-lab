import ref


def check(workdir):
    m = {"head_connected": 0.0}
    try:
        engine = ref.get_reference_engine()
        hidden, _ = engine.forward_target([10, 20])
        logits = engine.head.forward(hidden)
        if logits.shape == (2, 200):
            m["head_connected"] = 1.0
    except Exception:
        pass
    return m
