import hashlib

def _block0_hash(req):
    token_bytes = req["token"].encode("utf-8")
    salt_bytes = req["cache_salt"].to_bytes(4, byteorder="big", signed=True)
    return hashlib.sha256(token_bytes + salt_bytes).digest()[:8]

def grade(sol, fx) -> dict:
    # Test pairs: identical tokens with same/different salts,
    # different tokens, negative salts.
    test_pairs = [
        ({"token": "abc", "cache_salt": 1}, {"token": "abc", "cache_salt": 1}),
        ({"token": "abc", "cache_salt": 1}, {"token": "abc", "cache_salt": 2}),
        ({"token": "xyz", "cache_salt": 0}, {"token": "xyz", "cache_salt": 0}),
        ({"token": "xyz", "cache_salt": -1}, {"token": "xyz", "cache_salt": -1}),
        ({"token": "foo", "cache_salt": 12345}, {"token": "bar", "cache_salt": 12345}),
    ]

    ok = 1.0
    try:
        fn = getattr(sol, "blocks_collide")
    except AttributeError:
        return {"exact_match": 0.0}

    for req1, req2 in test_pairs:
        try:
            got = bool(fn(req1, req2))
        except Exception:
            ok = 0.0
            break

        ref = _block0_hash(req1) == _block0_hash(req2)
        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
