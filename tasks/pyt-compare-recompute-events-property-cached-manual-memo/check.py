N = 47
K = 5


def _ref_expensive(n):
    total = 0
    for i in range(n):
        total += i * i
    return total


def grade(sol, fx) -> dict:
    try:
        orig_expensive = sol.expensive
    except AttributeError:
        return {"exact_match": 0.0}

    for n in (0, 1, 5, N):
        try:
            got = orig_expensive(n)
        except Exception:
            return {"exact_match": 0.0}
        if got != _ref_expensive(n):
            return {"exact_match": 0.0}

    ref_value = _ref_expensive(N)
    counts = {}

    def make_counter(key):
        def wrapper(*args, **kwargs):
            counts[key] = counts.get(key, 0) + 1
            return orig_expensive(*args, **kwargs)
        return wrapper

    demo_specs = [
        ("property", "PropertyDemo"),
        ("cached_property", "CachedPropertyDemo"),
        ("manual_memo", "ManualMemoDemo"),
    ]

    result_counts = []
    try:
        for key, cls_name in demo_specs:
            counts[key] = 0
            sol.expensive = make_counter(key)
            cls = getattr(sol, cls_name)
            obj = cls(N)
            last_val = None
            for _ in range(K):
                last_val = obj.value
            if last_val != ref_value:
                return {"exact_match": 0.0}
            result_counts.append(counts[key])
    except Exception:
        return {"exact_match": 0.0}
    finally:
        sol.expensive = orig_expensive

    expected_counts = [K, 1, 1]
    ok = 1.0 if result_counts == expected_counts else 0.0
    return {"exact_match": ok}
