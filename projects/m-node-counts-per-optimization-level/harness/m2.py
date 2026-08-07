import ref


def check(workdir):
    from ortopt.fusions import tracked_fusions

    matched = 0
    levels = [0, 1, 99]
    for m in ref.MODELS:
        for lvl in levels:
            want = ref.tracked_fusions(m, lvl)
            got = tracked_fusions(m, lvl)
            if sorted(got) == sorted(want):
                matched += 1
    return {"fusions_matched": float(matched)}
