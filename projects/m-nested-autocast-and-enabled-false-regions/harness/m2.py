import ref


def check(workdir):
    from autocast.analyzer import analyze_regions
    trace = [
        {"event": "push", "enabled": True, "dtype": "float16"},
        {"event": "op", "sensitive": False, "used_dtype": "float16"},
        {"event": "push", "enabled": False, "dtype": "float16"},
        {"event": "op", "sensitive": False, "used_dtype": "float16"},
        {"event": "pop"},
        {"event": "pop"}
    ]
    res = analyze_regions(trace)
    violations = res.get("violations", -1)
    correct = res.get("correct", -1)
    out = {"violations_match": 1.0 if violations == 1 else 0.0, "precision_correct": 1.0 if correct == 1 else 0.0}
    return out
