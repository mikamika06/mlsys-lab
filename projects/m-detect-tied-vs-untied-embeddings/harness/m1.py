import ref


def check(workdir):
    from ggufmap.embed import check_tied

    cases_m1, _, _ = ref.get_cases()
    ok = 0
    for sd, expected in cases_m1:
        try:
            res = check_tied(sd)
            if res == expected:
                ok += 1
        except Exception:
            pass
    return {"embed_matched": 1.0 if ok == len(cases_m1) else 0.0}
