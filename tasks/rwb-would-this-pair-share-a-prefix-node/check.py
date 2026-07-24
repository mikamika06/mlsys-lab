def grade(sol, fx) -> dict:
    import random, string
    ok = 1.0
    for _ in range(20):
        len_a = random.randint(0, 5)
        len_b = random.randint(0, 5)
        a = [random.choice(string.ascii_lowercase) for _ in range(len_a)]
        b = [random.choice(string.ascii_lowercase) for _ in range(len_b)]
        try:
            got = sol.shares_prefix(a, b)
        except Exception:
            ok = 0.0
            break
        expected = (len_a > 0 and len_b > 0 and a[0] == b[0])
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
