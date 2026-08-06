import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from shapes.verifier import propagate_shapes as student_fn
    except ImportError:
        return {"exact_match": 0.0}

    cases = [
        (
            {"s1": (2, "s0"), "s2": (3, "s1")},
            {"x": [(1, "s1"), (1, "s0")]},
            [
                {"in": "x", "out": "y", "shape": [(1, "s0"), (2, None), (-1, None)]},
                {"in": "y", "out": "z", "shape": [(-1, None), (1, "s0")]}
            ]
        ),
        (
            {"batch": (4, "micro_batch")},
            {"a": [(1, "batch"), (128, None)]},
            [
                {"in": "a", "out": "b", "shape": [(1, "micro_batch"), (-1, None)]}
            ]
        )
    ]

    ok = 0
    for c, i, o in cases:
        want = ref.propagate_shapes(o, i, c)
        try:
            got = student_fn(o, i, c)
            if want == got:
                ok += 1
        except Exception:
            pass

    return {"exact_match": float(ok) / len(cases)}
