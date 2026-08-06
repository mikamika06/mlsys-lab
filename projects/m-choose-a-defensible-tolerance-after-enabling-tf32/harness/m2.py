import ref


def check(workdir):
    from tf32guard.tolerance import suggest_tolerance

    shapes = [(64, 64), (128, 256), (512, 512)]
    conds = [1.0, 10.0, 100.0]
    match_count = 0
    safe_count = 0
    total = len(shapes) * len(conds)

    for shape in shapes:
        for cond in conds:
            want = ref.reference_tolerance(shape, cond)
            try:
                got = float(suggest_tolerance(shape, cond))
            except Exception:
                got = -1.0
            if abs(got - want) < 1e-5:
                match_count += 1
            if got >= want * 0.99:
                safe_count += 1

    return {
        "tolerance_match": 1.0 if match_count == total else 0.0,
        "safe_bound": 1.0 if safe_count == total else 0.0,
    }
