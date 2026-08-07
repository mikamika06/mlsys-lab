import ref


def check(workdir):
    from toolutils.normalize import normalize_messages

    out = {"messages_matched": 0.0}
    want = ref.normalize_messages(ref.MESSAGES_SAMPLE)
    got = normalize_messages(ref.MESSAGES_SAMPLE)

    if got == want:
        out["messages_matched"] = 1.0
    else:
        out["_note"] = f"expected {want}, got {got}"
    return out
