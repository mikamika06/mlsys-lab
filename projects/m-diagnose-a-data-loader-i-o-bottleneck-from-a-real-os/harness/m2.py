import ref


def check(workdir):
    from dataloader.diagnose import find_io_bottleneck

    cases = ref.get_test_cases()
    ok = 0
    for _, osrt in cases:
        want = max(osrt, key=lambda x: x["total_time_ms"])["name"]
        got = find_io_bottleneck(osrt)
        if got == want:
            ok += 1

    return {"bottleneck_identified": 1.0 if ok == len(cases) else 0.0}
