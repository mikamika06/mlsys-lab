import ref


def check(workdir):
    from kvcache.flash import requires_flash_attention

    out = {"flash_matched": 0.0}
    types = ["Q4_0", "Q4_K", "Q8_0", "F16", "F32", "IQ4_NL"]
    for t in types:
        want = ref.requires_flash_attention(t)
        try:
            got = requires_flash_attention(t)
        except Exception:
            got = not want
        if got != want:
            out["_note"] = f"type {t}: got {got}, want {want}"
            return out
    out["flash_matched"] = 1.0
    return out
