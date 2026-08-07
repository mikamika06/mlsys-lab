import ref


def check(workdir):
    m = {"memory_saved": 0.0}
    try:
        engine = ref.get_reference_engine()
        mem = engine.memory_usage_bytes()
        if mem["head_bytes"] < mem["separate_model_bytes"] * 0.1:
            m["memory_saved"] = 1.0
    except Exception:
        pass
    return m
