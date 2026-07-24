from mlsys import scorers


def _oracle(values, width):
    encoded = []
    decoded = []
    for value in values:
        b = int(value).to_bytes(width, byteorder="little", signed=False)
        encoded.append(b)
        decoded.append(int.from_bytes(b, byteorder="little", signed=False))
    if decoded != [int(v) for v in values]:
        raise AssertionError("oracle round trip failed")
    return encoded


def _flatten_bytes(items):
    return b"".join(items)


def grade(sol, fx) -> dict:
    cases = [
        ([0, 1, 255, 256], 2),
        ([1, 65535, 2**24 + 3], 4),
        ([0, 5, 42, 999999], 8),
        ([2**31, 2**32 + 7], 8),
    ]

    ref_all = []
    got_all = []
    for values, width in cases:
        try:
            ref = _oracle(values, width)
            got = sol.int_bytes_round_trip(list(values), width)
        except Exception:
            return {"byte_exact_fraction": 0.0}

        if not isinstance(got, list) or len(got) != len(ref):
            return {"byte_exact_fraction": 0.0}

        ref_all.extend(ref)
        got_all.extend(got)

    try:
        score = scorers.byte_exact_fraction(_flatten_bytes(ref_all), _flatten_bytes(got_all))
    except Exception:
        score = 0.0
    return {"byte_exact_fraction": score}
