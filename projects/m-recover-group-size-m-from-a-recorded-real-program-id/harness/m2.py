import ref

def check(workdir):
    from grouptrace.recover import recover_group_size
    ok = 0
    total = len(ref.TEST_CASES)
    for case in ref.TEST_CASES:
        trace = ref.generate_trace(case["grid_m"], case["grid_n"], case["group_size_m"])
        g = recover_group_size(trace)
        if g == case["group_size_m"]:
            ok += 1
    return {"group_size_matched": 1.0 if ok == total else 0.0}
