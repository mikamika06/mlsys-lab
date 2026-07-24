import hashlib


def _oracle(chunks, store):
    previous = b""
    found = set()
    for i, chunk in enumerate(chunks):
        previous = hashlib.sha256(previous + chunk).digest()
        if previous in store:
            for pos in store[previous]:
                found.add((i, pos))
    return sorted(found)


def _make_store(chunks, positions):
    store = {}
    previous = b""
    for i, chunk in enumerate(chunks):
        previous = hashlib.sha256(previous + chunk).digest()
        store.setdefault(previous, []).append(positions[i])
    return store


def grade(sol, fx) -> dict:
    base_cases = [
        [b"a", b"b", b"c"],
        [b"alpha", b"beta", b"gamma", b"delta"],
        [b"", b"x", b"x", b"y"],
    ]

    cases = []
    for chunks in base_cases:
        positions = [100 + i * 7 for i in range(len(chunks))]
        store = _make_store(chunks, positions)
        cases.append((chunks, store))

    chunks = [b"a", b"b", b"c"]
    store = _make_store([b"z", b"b", b"c"], [1, 2, 3])
    cases.append((chunks, store))

    chunks = [b"one", b"two", b"three"]
    store = _make_store(chunks, [5, 5, 8])
    cases.append((chunks, store))

    ok = 1.0
    for chunks, store in cases:
        try:
            got = sol.lookup_reused_chunks(chunks, store)
            got = sorted(set(tuple(x) for x in got))
        except Exception:
            ok = 0.0
            break

        if got != _oracle(chunks, store):
            ok = 0.0
            break

    return {"exact_match": ok}
