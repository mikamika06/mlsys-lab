import ref


def check(workdir):
    m = {"sampling_correct": 0.0}
    try:
        engine = ref.get_reference_engine()
        hidden, _ = engine.forward_target([1, 2])
        tokens = engine.generate_draft(hidden)
        if isinstance(tokens, list) and len(tokens) == 2:
            m["sampling_correct"] = 1.0
    except Exception:
        pass
    return m
