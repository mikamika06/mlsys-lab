import ref

def check(workdir):
    m = {"checkpoint_active": 0.0, "memory_reduction_ok": 0.0}
    try:
        from qlora_fix.memory import apply_checkpointing
        class DummyModel:
            pass
        mod = DummyModel()
        res = apply_checkpointing(mod)
        if res == ref.oracle_apply_checkpointing(mod) and getattr(mod, "checkpointing", False):
            m["checkpoint_active"] = 1.0
            m["memory_reduction_ok"] = 1.0
    except Exception:
        pass
    return m
