import ref


def check(workdir):
    from benchrunner.benchmark import attribute_runner_delta

    out = {"benchmarks_matched": 0.0}
    ok = 0
    for i, case in enumerate(ref.BENCH_CASES):
        want = ref.attribute_runner_delta(case)
        got = attribute_runner_delta(case)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    if ok == len(ref.BENCH_CASES):
        out["benchmarks_matched"] = 1.0
    return out
