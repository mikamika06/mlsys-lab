def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    m = {"traffic_weights_valid": 0.0, "step_shift_correct": 0.0}
    try:
        pol = ref.get_reference_policy()
        w0 = pol.get_weight(0)
        w2 = pol.get_weight(2)
        if 0.0 <= w0 <= 1.0 and 0.0 <= w2 <= 1.0:
            m["traffic_weights_valid"] = 1.0
        if w2 > w0:
            m["step_shift_correct"] = 1.0
    except Exception:
        pass
    return m
