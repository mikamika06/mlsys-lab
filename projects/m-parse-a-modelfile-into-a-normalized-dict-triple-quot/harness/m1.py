import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from modelfile.parser import parse
    except ImportError:
        return {"fixtures_matched": 0.0}

    ok = 0
    out = {"fixtures_matched": 0.0}
    for i, text in enumerate(ref.FIXTURES):
        try:
            got = parse(text)
            want = ref.parse(text)
            if got == want:
                ok += 1
            else:
                if "_note" not in out:
                    out["_note"] = f"fixture {i}: got {got}, want {want}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"fixture {i} error: {e}"

    out["fixtures_matched"] = float(ok)
    return out
