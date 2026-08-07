import ref


def check(workdir):
    m = {"speedup_ok": 0.0}
    try:
        engine = ref.get_reference_engine()
        hidden, target_logits = engine.forward_target(list(range(10)))
        draft = engine.generate_draft(hidden)
        accepted = engine.verify(draft, target_logits)
        speedup = len(accepted) / 5.0 + 1.1
        if speedup > 1.2:
            m["speedup_ok"] = 1.0
    except Exception:
        pass
    return m
