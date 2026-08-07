import ref

def check(workdir):
    from dnnlog import parse_row
    out = {"rows_matched": 0.0}
    ok = 0
    for line in ref.LOGS:
        parsed = parse_row(line)
        if parsed and "primitive" in parsed and "time_ms" in parsed:
            ok += 1
    out["rows_matched"] = float(ok)
    return out
