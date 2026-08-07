import ref

def check(workdir):
    from gguf_parser.parser import compute_overhead

    out = {"overheads_matched": 0.0, "total": float(len(ref.FIXTURES))}
    ok = 0
    for i, data in enumerate(ref.FIXTURES):
        manifest = ref.parse_header(data)
        want = ref.compute_overhead(manifest)
        try:
            got = compute_overhead(manifest)
            if got == want:
                ok += 1
            else:
                out["_note"] = f"fixture {i} mismatch: got {got}, want {want}"
        except Exception as e:
            out["_note"] = f"fixture {i} error: {e}"
    out["overheads_matched"] = float(ok)
    return out
