import ref


def check(workdir):
    from modelfile.parser import parse_modelfile

    out = {"parsed_matched": 0.0}
    ok = 0
    for content, expected in ref.SAMPLES:
        got = parse_modelfile(content)
        if got == expected:
            ok += 1
        else:
            out["_note"] = f"got {got}, expected {expected}"
            break
    if ok == len(ref.SAMPLES):
        out["parsed_matched"] = 1.0
    return out
