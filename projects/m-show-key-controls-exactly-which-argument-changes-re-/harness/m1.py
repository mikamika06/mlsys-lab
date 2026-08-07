import ref


def check(workdir):
    from triton_tune.autokey import check_key_triggers
    out = {"keys_matched": 0.0}
    try:
        want = check_key_triggers(ref.KEY_ARGS, ref.CALL_SEQUENCES)
        got = check_key_triggers(ref.KEY_ARGS, ref.CALL_SEQUENCES)
        if got == want:
            out["keys_matched"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"Exception: {type(e).__name__}: {str(e)[:120]}"
    return out
