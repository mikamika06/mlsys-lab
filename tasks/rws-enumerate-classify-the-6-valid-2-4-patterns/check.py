def grade(sol, fx) -> dict:
    patterns = [tuple(int(b) for b in format(i, '04b')) for i in range(16)]
    valid = sorted([p for p in patterns if sum(p) == 2])
    ref_map = {p: idx for idx, p in enumerate(valid)}

    test_vectors = [list(p) for p in patterns]
    test_vectors.extend([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
    ])

    expected = [ref_map.get(tuple(vec), -1) for vec in test_vectors]

    try:
        got = sol.classify_patterns(test_vectors)
    except Exception:
        return {"exact_match": 0.0}

    if hasattr(got, "tolist"):
        got = got.tolist()

    if got == expected:
        return {"exact_match": 1.0}
    return {"exact_match": 0.0}
