import ref


def check(workdir):
    from roofline.predict import compute_decode_bytes

    out = {"bytes_matched": 0.0}
    cfg = ref.CONFIGS[0]
    batch_size = 4
    context_len = 128
    want = ref.compute_decode_bytes(cfg, batch_size, context_len)
    try:
        got = compute_decode_bytes(cfg, batch_size, context_len)
    except Exception:
        got = -1
    if got == want:
        out["bytes_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, reference {want}"
    return out
