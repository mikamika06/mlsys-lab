import ref

def check(workdir):
    from onednn.parser import parse_row
    out = {"rows_matched": 0.0}
    matched = 0
    for line, want in zip(ref.SAMPLE_LOGS, ref.EXPECTED_PARSED):
        got = parse_row(line)
        if got == want:
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"line {line[:30]}: got {got}, want {want}"
    out["rows_matched"] = float(matched)
    return out
