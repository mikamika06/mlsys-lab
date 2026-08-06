import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from shapes.verifier import confirm_relationship as student_fn
    except ImportError:
        return {"exact_match": 0.0, "_note": "ImportError"}

    constraints = {
        "s1": (2, "s0"),
        "s2": (3, "s1"),
        "s3": (4, "s0"),
        "s4": (1, "s2")
    }
    pairs = [
        ("s1", "s0"),
        ("s2", "s0"),
        ("s4", "s0"),
        ("s3", "s0"),
        ("s4", "s1"),
        ("s0", "s1"),
        ("s5", "s0")
    ]
    ok = 0
    for d2, d1 in pairs:
        want = ref.confirm_relationship(constraints, d2, d1)
        got = student_fn(constraints, d2, d1)
        if want == got:
            ok += 1

    return {"exact_match": float(ok) / len(pairs)}
