import ref


def check(workdir):
    from trtprof.profile import select_opt_shape, calculate_profile_bounds

    fixtures = ref.make_workload_fixtures()
    out = {"distributions_matched": 0.0}
    matched = 0

    for fix in fixtures:
        prof_spec = fix["profiles"][0]
        strat = prof_spec["strategy"]
        samples = prof_spec["tensors"]["input_ids"]["samples"]

        ref_opt = ref.select_opt_shape(samples, strategy=strat)
        ref_bounds = ref.calculate_profile_bounds(samples, strategy=strat, padding_ratio=0.1)

        got_opt = select_opt_shape(samples, strategy=strat)
        got_bounds = calculate_profile_bounds(samples, strategy=strat, padding_ratio=0.1)

        if got_opt == ref_opt and got_bounds == ref_bounds:
            matched += 1

    out["distributions_matched"] = float(matched)
    return out
