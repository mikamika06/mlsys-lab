import ref

def check(workdir):
    from vllmlog.parser import parse_log
    ok = 0
    for log in ref.LOGS:
        want = ref.parse_log(log)
        got = parse_log(log)
        if got == want:
            ok += 1
    return {"parsed_match": 1.0 if ok == len(ref.LOGS) else 0.0}
