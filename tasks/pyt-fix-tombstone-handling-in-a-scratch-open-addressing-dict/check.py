"""Grade a from-scratch open-addressing dict against a real Python dict,
run through a fixed sequence of set/get/delete/contains calls built around
deliberately colliding integer keys (capacity=8, keys 3/11/19/27/43 all hash
to bucket 3), interior deletes, reinsertion into a tombstone, consecutive
tombstones, and deleting the first key in a chain.
"""

_OPS = [
    ("set", 3, "a"),
    ("set", 11, "b"),
    ("set", 19, "c"),
    ("set", 27, "d"),
    ("get", 3, None),
    ("get", 11, None),
    ("get", 19, None),
    ("get", 27, None),
    ("delete", 11, None),
    ("get", 19, None),
    ("get", 27, None),
    ("contains", 11, None),
    ("delete", 19, None),
    ("get", 27, None),
    ("set", 11, "b2"),
    ("get", 11, None),
    ("contains", 19, None),
    ("delete", 3, None),
    ("get", 27, None),
    ("get", 11, None),
    ("set", 43, "e"),
    ("get", 43, None),
    ("get", 27, None),
    ("delete", 27, None),
    ("contains", 27, None),
    ("get", 999, None),
]


def grade(sol, fx) -> dict:
    try:
        d = sol.ScratchDict(capacity=8)
    except Exception:
        return {"exact_match": 0.0}

    ref: dict = {}

    for op, key, val in _OPS:
        try:
            if op == "set":
                d.set(key, val)
                ref[key] = val

            elif op == "get":
                should_exist = key in ref
                try:
                    got = d.get(key)
                    if not should_exist or got != ref[key]:
                        return {"exact_match": 0.0}
                except KeyError:
                    if should_exist:
                        return {"exact_match": 0.0}

            elif op == "delete":
                should_exist = key in ref
                try:
                    d.delete(key)
                    if not should_exist:
                        return {"exact_match": 0.0}
                    del ref[key]
                except KeyError:
                    if should_exist:
                        return {"exact_match": 0.0}

            elif op == "contains":
                got = key in d
                if bool(got) != (key in ref):
                    return {"exact_match": 0.0}
        except Exception:
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
