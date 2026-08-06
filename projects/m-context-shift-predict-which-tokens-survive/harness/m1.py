import ref

def check(workdir):
    try:
        from caching.prefix import longest_surviving_prefix
    except ImportError:
        return {"exact_match": 0.0, "_note": "Could not import longest_surviving_prefix"}

    import random
    rng = random.Random(42)

    ok = 0
    total = 100
    for _ in range(total):
        cached = []
        for _ in range(4):
            cached.append([rng.randint(1, 5) for _ in range(rng.randint(5, 15))])

        target = rng.choice(cached)
        cut = rng.randint(0, len(target))
        new_prompt = target[:cut] + [rng.randint(6, 10) for _ in range(rng.randint(2, 5))]

        want = ref.longest_surviving_prefix(cached, new_prompt)
        try:
            got = longest_surviving_prefix(cached, new_prompt)
            if got == want:
                ok += 1
        except Exception:
            pass

    return {"exact_match": ok / total}
