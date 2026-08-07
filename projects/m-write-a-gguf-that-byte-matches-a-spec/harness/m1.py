import ref

def check(workdir):
    from gguf_spec.writer import write_gguf
    spec = ref.make_spec()
    want = ref.write_gguf(spec)
    got = write_gguf(spec)
    matched = 1.0 if got == want else 0.0
    out = {"bytes_matched": matched}
    if matched == 0.0:
        out["_note"] = f"got len {len(got) if got else 0}, want len {len(want)}"
    return out
