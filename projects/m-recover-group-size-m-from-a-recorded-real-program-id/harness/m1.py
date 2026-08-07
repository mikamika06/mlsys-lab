import ref

def check(workdir):
    from grouptrace.parser import parse_trace
    ok = 0
    total = len(ref.TEST_CASES)
    for case in ref.TEST_CASES:
        trace = ref.generate_trace(case["grid_m"], case["grid_n"], case["group_size_m"])
        gm, gn = parse_trace(trace)
        if gm == case["grid_m"] and gn == case["grid_n"]:
            ok += 1
    return {"dimensions_matched": 1.0 if ok == total else 0.0}
