import ref


def check(workdir):
    from tvmutils.validator import verify_schedule_behavior

    out = {"schedule_verified": 0.0}
    ok = 0
    for tc in ref.TEST_CASES:
        try:
            res = verify_schedule_behavior(tc["extent"], tc["factor"])
            if isinstance(res, dict) and "outer" in res and "inner" in res:
                if res["outer"] * res["inner"] >= tc["extent"]:
                    ok += 1
        except Exception:
            pass
    if ok == len(ref.TEST_CASES):
        out["schedule_verified"] = 1.0
    return out
