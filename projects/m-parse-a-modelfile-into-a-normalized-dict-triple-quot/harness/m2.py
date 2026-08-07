import ref


def check(workdir):
    from modelfile.parser import parse_modelfile
    from modelfile.emitter import emit_modelfile

    out = {"roundtrip_matched": 0.0}
    ok = 0
    for content, _ in ref.SAMPLES:
        parsed1 = parse_modelfile(content)
        emitted = emit_modelfile(parsed1)
        parsed2 = parse_modelfile(emitted)
        if parsed1 == parsed2:
            ok += 1
        else:
            out["_note"] = f"Roundtrip failed for content"
            break
    if ok == len(ref.SAMPLES):
        out["roundtrip_matched"] = 1.0
    return out
