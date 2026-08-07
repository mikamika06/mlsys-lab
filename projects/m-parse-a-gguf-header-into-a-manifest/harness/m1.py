import ref

def check(workdir):
    from gguf_parser.parser import parse_header

    out = {"manifests_matched": 0.0, "total": float(len(ref.FIXTURES))}
    ok = 0
    for i, data in enumerate(ref.FIXTURES):
        want = ref.parse_header(data)
        try:
            got = parse_header(data)
            if got.get("metadata") == want["metadata"] and got.get("tensors") == want["tensors"]:
                ok += 1
            else:
                out["_note"] = f"fixture {i} mismatch"
        except Exception as e:
            out["_note"] = f"fixture {i} error: {e}"
    out["manifests_matched"] = float(ok)
    return out
