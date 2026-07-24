import gc


def grade(sol, fx) -> dict:
    try:
        class Key:
            def __init__(self, tag):
                self.tag = tag

        cache = sol.IdentityCache()
        keys = [Key(i) for i in range(6)]
        for i in range(6):
            cache.put(keys[i], f"value-{i}")

        if len(cache) != 6:
            return {"exact_match": 0.0}

        to_kill = [1, 3, 4]
        survivors_idx = [i for i in range(6) if i not in to_kill]

        for i in to_kill:
            keys[i] = None
        gc.collect()

        if len(cache) != len(survivors_idx):
            return {"exact_match": 0.0}

        for i in survivors_idx:
            if cache.get(keys[i]) != f"value-{i}":
                return {"exact_match": 0.0}

        missing_key = Key(-1)
        if cache.get(missing_key, "sentinel") != "sentinel":
            return {"exact_match": 0.0}

        for i in survivors_idx:
            keys[i] = None
        gc.collect()

        if len(cache) != 0:
            return {"exact_match": 0.0}

    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0}
