import ref


def check(workdir):
    from shm.config import reconstruct_configs

    cases = ref.get_m2_cases()
    ok = 0
    for case in cases:
        got = reconstruct_configs(
            case["error_msg"], case["max_bytes"], case["candidates"]
        )
        want = case["expected"]
        if got == want:
            ok += 1
    score = float(ok) / float(len(cases)) if cases else 1.0
    return {"match_score": score}
