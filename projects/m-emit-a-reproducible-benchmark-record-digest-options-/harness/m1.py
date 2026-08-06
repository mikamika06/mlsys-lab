import ref


def check(workdir):
    from benchmark.recorder import emit_record

    out = {"record_matched": 0.0}
    ok = 0
    for run in ref.RUNS:
        want = ref.generate_record(run)
        got = emit_record(run["options"], run["phases"])
        if got == want:
            ok += 1
    if ok == len(ref.RUNS):
        out["record_matched"] = 1.0
    return out
