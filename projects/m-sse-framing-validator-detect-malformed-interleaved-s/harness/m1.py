import ref

def check(workdir):
    from ssevall.validator import validate_sse_stream

    out = {"streams_validated": 0.0}
    ok = 0
    for chunks, expected in ref.STREAMS:
        try:
            got = validate_sse_stream(chunks)
            if got == expected:
                ok += 1
        except Exception:
            pass
    out["streams_validated"] = float(ok)
    return out
