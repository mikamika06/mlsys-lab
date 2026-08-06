import ref


def check(workdir):
    from extractor.parser import extract_stop_sequences
    out = {"stops_matched": 0.0}
    ok = True
    for cfg in ref.MODELS:
        want = ref.extract_stop_sequences(cfg)
        got = extract_stop_sequences(cfg)
        if got != want:
            ok = False
            out["_note"] = f"expected {want}, got {got}"
            break
    if ok:
        out["stops_matched"] = 1.0
    return out
