import w


def check(workdir):
    out = {"magic_ok": 0.0, "version_ok": 0.0, "counts_ok": 0.0, "rejects_garbage": 0.0}
    if w.gguf_or_none() is None:
        return w.needs_gguf(out)
    from gguf_lab import read_header
    import os

    tmp = w.tmpdir()
    p = w.fixture(tmp)
    h = read_header(p)
    out["magic_ok"] = 1.0 if h.get("magic") == "GGUF" else 0.0
    out["version_ok"] = 1.0 if h.get("version") == 3 else 0.0
    out["counts_ok"] = 1.0 if (h.get("tensor_count") == 2 and h.get("kv_count") == 4) else 0.0

    bad = os.path.join(tmp, "bad.bin")
    with open(bad, "wb") as f:
        f.write(b"NOPE" + b"\x00" * 64)
    try:
        read_header(bad)
        out["rejects_garbage"] = 0.0
        out["_note"] = "a file whose magic is not GGUF was accepted"
    except Exception:
        out["rejects_garbage"] = 1.0
    return out
