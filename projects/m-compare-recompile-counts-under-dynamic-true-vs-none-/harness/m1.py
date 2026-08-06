import ref

def check(workdir):
    from recompiles.analysis import compare_recompiles
    cases = ref.get_test_cases()
    modes = [True, False, None]
    ok = 0
    total = 0
    for shapes in cases:
        for mode in modes:
            total += 1
            want = _ref_comp(shapes, mode)
            got = compare_recompiles(shapes, mode)
            if got == want:
                ok += 1
    return {"comparison_match": 1.0 if ok == total else 0.0}

def _ref_comp(shapes, dynamic_mode):
    seen = set()
    recompiles = 0
    for shape in shapes:
        if dynamic_mode is True:
            key = tuple(1 if x > 1 else x for x in shape)
        elif dynamic_mode is False:
            key = shape
        else:
            key = tuple(x if x <= 4 else -1 for x in shape)
        if key not in seen:
            seen.add(key)
            recompiles += 1
    return recompiles
