import ref


def check(workdir):
    from profiler.diagnose import diagnose_zero_gpu_events
    from profiler.schedule import schedule_summary

    out = {"arithmetic_matched": 0.0, "diagnosis_matched": 0.0}

    arith_ok = 0
    for i, tc in enumerate(ref.ARITHMETIC_TEST_CASES):
        want = ref.ref_schedule_summary(**tc)
        got = schedule_summary(**tc)
        if got == want:
            arith_ok += 1
        elif "_note" not in out:
            out["_note"] = f"arithmetic case {i}: got {got}, want {want}"

    if arith_ok == len(ref.ARITHMETIC_TEST_CASES):
        out["arithmetic_matched"] = 1.0

    diag_ok = 0
    for i, tc in enumerate(ref.DIAGNOSE_TEST_CASES):
        want = tc["expected"]
        got = diagnose_zero_gpu_events(tc["config"])
        if got == want:
            diag_ok += 1
        elif "_note" not in out:
            out["_note"] = f"diagnosis case {i}: got '{got}', want '{want}'"

    if diag_ok == len(ref.DIAGNOSE_TEST_CASES):
        out["diagnosis_matched"] = 1.0

    return out
