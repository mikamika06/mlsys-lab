def check(workdir):
    from speculative.alignment import align_bytes
    m = {"alignment_ok": 0.0}
    try:
        vocab = [b"a", b"b"]
        res = align_bytes([98, 97], vocab)
        if res == [1, 0]:
            m["alignment_ok"] = 1.0
    except Exception:
        pass
    return m
