import ref

def check(workdir):
    from rawstream.reassemble import reassemble_stream
    ok = 0
    streams = ref.STREAMS
    for s in streams:
        want = ref.reassemble_stream(s)
        try:
            got = reassemble_stream(s)
        except Exception:
            got = None
        if got == want:
            ok += 1
    return {"streams_matched": float(ok)}
