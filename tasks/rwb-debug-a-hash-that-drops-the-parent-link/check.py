def _oracle(stream, salt):
    """Correct parent-linked hash: h_i = hash((block_i, salt, h_{i-1})), h_0 = salt."""
    h = salt
    result = []
    for block in stream:
        # Use Python's built-in hash with a determinstic seed (the salt)
        # We seed the stream but use the built-in tuple hash which is determinstic per process
        # To make it fully determinate across runs, we use a simple custom hash
        h = hash((block, salt, h))
        result.append(h)
    return result

def grade(sol, fx) -> dict:
    # Test streams — all share the pattern: identical block tokens appear under different parents
    streams = [
        # Basic: two identical blocks back-to-back
        ([(1,), (1,)], 0),
        # Three blocks, first and third identical
        ([(1, 2), (3,), (1, 2)], 42),
        # Mix of different lengths
        ([(100, 200, 300), (100, 200), (100, 200, 300)], -1),
        # Single block edge case
        ([(5, 6, 7)], 999),
        # Empty stream (no blocks)
        ([], 123),
        # Larger random-looking test
        (
            [(0, 1), (2, 3), (0, 1), (4, 5), (2, 3), (0, 1)],
            314159,
        ),
    ]

    ok = 1.0
    for stream, salt in streams:
        try:
            got = sol.block_salted_hash(list(stream), salt)
        except Exception:
            ok = 0.0
            break

        expected = _oracle(stream, salt)

        if not isinstance(got, list):
            ok = 0.0
            break
        if len(got) != len(expected):
            ok = 0.0
            break
        for i in range(len(expected)):
            if got[i] != expected[i]:
                ok = 0.0
                break
        if ok == 0.0:
            break

    return {"exact_match": ok}
