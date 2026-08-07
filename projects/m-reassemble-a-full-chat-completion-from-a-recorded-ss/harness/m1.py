import ref


def check(workdir):
    from chatparse.stream import reassemble_chat_completion

    out = {"streams_matched": 0.0}
    ok = 0
    for sample in ref.STREAM_SAMPLES:
        want = ref.reassemble_stream(sample)
        got = reassemble_chat_completion(sample)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {got}, want {want}"
    out["streams_matched"] = float(ok)
    return out
