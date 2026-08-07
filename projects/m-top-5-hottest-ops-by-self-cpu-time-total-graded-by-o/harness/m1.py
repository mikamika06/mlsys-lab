import ref

def check(workdir):
    from profops.parser import parse_profiler_table
    out = {"parsed_correctly": 0.0}
    ok = 0
    for rows in ref.PROFILES:
        want = ref.parse_table(rows)
        got = parse_profiler_table(rows)
        if got == want:
            ok += 1
    if ok == len(ref.PROFILES):
        out["parsed_correctly"] = 1.0
    return out
