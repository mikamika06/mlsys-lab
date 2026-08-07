import ref


def check(workdir):
    from dsengine.overlap import compute_speedup, compute_step_time, min_overlap_for_speedup

    out = {"overlap_cases_matched": 0.0, "speedup_bounds_valid": 0.0}
    ok_cases = 0
    total_cases = len(ref.OVERLAP_CASES)

    for case in ref.OVERLAP_CASES:
        tc, tg, factor, target_s = case
        want_time = ref.compute_step_time(tc, tg, factor)
        want_speedup = ref.compute_speedup(tc, tg, factor)
        want_min_factor = ref.min_overlap_for_speedup(tc, tg, target_s)

        got_time = compute_step_time(tc, tg, factor)
        got_speedup = compute_speedup(tc, tg, factor)
        got_min_factor = min_overlap_for_speedup(tc, tg, target_s)

        time_ok = abs(got_time - want_time) < 1e-6
        speedup_ok = abs(got_speedup - want_speedup) < 1e-6
        min_factor_ok = abs(got_min_factor - want_min_factor) < 1e-6

        if time_ok and speedup_ok and min_factor_ok:
            ok_cases += 1
        elif "_note" not in out:
            out["_note"] = f"mismatch on ({tc}, {tg}, {factor}): time got {got_time} vs want {want_time}"

    if ok_cases == total_cases:
        out["overlap_cases_matched"] = 1.0
        out["speedup_bounds_valid"] = 1.0

    return out
