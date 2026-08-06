import ref


def check(workdir):
    from trtprof.planner import build_profile_plan

    fixtures = ref.make_workload_fixtures()
    out = {"plans_matched": 0.0, "bounds_valid": 0.0}

    all_matched = True
    bounds_valid = True

    for spec in fixtures:
        ref_plan = ref.build_profile_plan(spec)
        got_plan = build_profile_plan(spec)

        if got_plan != ref_plan:
            all_matched = False
            break

        for prof in got_plan:
            for t_name, bounds in prof.items():
                min_s, opt_s, max_s = bounds["min"], bounds["opt"], bounds["max"]
                for mi, op, ma in zip(min_s, opt_s, max_s):
                    if not (mi <= op <= ma):
                        bounds_valid = False

    if all_matched:
        out["plans_matched"] = 1.0
    if bounds_valid:
        out["bounds_valid"] = 1.0

    return out
