import ref


def check(workdir):
    """Check milestone 2."""
    from quantlib.scaling import decide_scaling_mode
    _, histories, _ = ref.get_test_cases()
    match_count = 0
    total = len(histories)
    for h in histories:
        ref_mode = ref.decide_scaling_mode(h)
        try:
            got_mode = decide_scaling_mode(h)
            if got_mode == ref_mode:
                match_count += 1
        except Exception:
            pass
    out = {"decision_matched": 1.0 if match_count == total else 0.0}
    return out
