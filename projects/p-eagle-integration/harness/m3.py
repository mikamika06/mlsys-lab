import ref


def check(workdir):
    m = {"acceptance_rate_ok": 0.0}
    try:
        engine = ref.get_reference_engine()
        hidden, target_logits = engine.forward_target([1, 2, 3])
        draft = engine.generate_draft(hidden)
        accepted = engine.verify(draft, target_logits, temperature=1.0)
        if len(accepted) > 0:
            m["acceptance_rate_ok"] = 1.0
    except Exception:
        pass
    return m
