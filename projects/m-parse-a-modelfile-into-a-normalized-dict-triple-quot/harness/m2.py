import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from modelfile.parser import dumps, parse
        from modelfile.stops import find_missing_stops
    except ImportError:
        return {"roundtrips": 0.0, "missing_found": 0.0}

    out = {"roundtrips": 0.0, "missing_found": 0.0}
    ok_rt = 0
    for i, text in enumerate(ref.FIXTURES):
        try:
            ast = ref.parse(text)
            dumped = dumps(ast)
            roundtripped = parse(dumped)
            if roundtripped == ast:
                ok_rt += 1
            else:
                if "_note" not in out:
                    out["_note"] = f"roundtrip {i} failed: got {roundtripped}, want {ast}"
        except Exception as e:
             if "_note" not in out:
                 out["_note"] = f"roundtrip {i} error: {e}"

    out["roundtrips"] = float(ok_rt)

    try:
        ast1 = ref.parse(ref.FIXTURES[0])
        miss1 = find_missing_stops(ast1)
        ast2 = ref.parse(ref.FIXTURES[1])
        miss2 = find_missing_stops(ast2)
        if set(miss1) == {"<|start_header_id|>", "<|end_header_id|>"} and set(miss2) == {"<|im_start|>"}:
            out["missing_found"] = 1.0
        else:
            if "_note" not in out:
                out["_note"] = f"stops mismatch. got {miss1}, {miss2}"
    except Exception as e:
        if "_note" not in out:
             out["_note"] = f"stops error: {e}"

    return out
