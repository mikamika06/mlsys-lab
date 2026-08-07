import ref


def check(workdir):
    from runner.parser import parse_truncated_log

    log = ref.generate_log_fixture()
    want = parse_truncated_log(log)

    try:
        from skeleton.runner.parser import parse_truncated_log as skeleton_parse
        got_skel = skeleton_parse(log)
    except Exception:
        got_skel = None

    try:
        got = parse_truncated_log(log)
    except Exception as e:
        return {"sequence_matched": 0.0, "_note": f"reference failed: {e}"}

    if got_skel == want:
        return {"sequence_matched": 0.0, "_note": "skeleton passes check"}

    matched = 1.0 if got == want else 0.0
    out = {"sequence_matched": matched}
    if matched == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
