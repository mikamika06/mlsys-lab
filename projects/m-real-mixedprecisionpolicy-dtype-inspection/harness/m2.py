import ref

def check(workdir):
    from mpinspect.sync import count_reduce_scatters
    scenarios = [
        (10, 4, True),
        (10, 4, False),
        (8, 2, True),
        (8, 1, False),
    ]
    ok = 0
    for num_steps, accum_steps, use_no_sync in scenarios:
        want = ref.count_reduce_scatters(num_steps, accum_steps, use_no_sync)
        got = count_reduce_scatters(num_steps, accum_steps, use_no_sync)
        if got == want:
            ok += 1
    return {"sync_matched": 1.0 if ok == len(scenarios) else 0.0}
