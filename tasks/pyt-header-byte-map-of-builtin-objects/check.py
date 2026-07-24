import sys


def _oracle(objects):
    header = 16
    return [
        [int(sys.getsizeof(obj)), header, int(sys.getsizeof(obj)) - header]
        for obj in objects
    ]


def grade(sol, fx) -> dict:
    cases = [
        [
            0,
            123456789,
            3.5,
            "abc",
            "",
            (1, 2),
            [],
            [1, 2, 3],
            b"abc",
            True,
            None,
        ],
        [
            -1,
            2.0,
            "longer text",
            (None, False, 7),
            bytearray(b"xy"),
        ],
    ]

    ok = 1.0
    for objects in cases:
        expected = _oracle(objects)
        try:
            got = sol.header_byte_map(objects)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

        if not all(
            isinstance(x, int)
            for row in got
            for x in row
        ):
            ok = 0.0
            break

    return {"exact_match": ok}
