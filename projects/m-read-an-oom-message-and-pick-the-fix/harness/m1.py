import ref


def check(workdir):
    from triage.parser import parse_oom_message, pick_fix

    out = {"oom_parsed": 0.0}
    for msg in ref.SAMPLE_OOMS:
        parsed = parse_oom_message(msg)
        if not isinstance(parsed, dict) or "requested_bytes" not in parsed:
            out["_note"] = f"parse_oom_message failed on: {msg[:60]}"
            return out
        fix = pick_fix(msg)
        if fix not in ("set_max_split_size", "empty_cache", "reduce_batch_size"):
            out["_note"] = f"pick_fix returned invalid action: {fix}"
            return out

    out["oom_parsed"] = 1.0
    return out
