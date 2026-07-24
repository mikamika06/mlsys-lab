def _oracle(strings):
    encodings = ["utf-8", "utf-16-le", "latin-1"]
    result = {}
    for enc in encodings:
        result[enc] = []
        for s in strings:
            b = s.encode(enc)
            result[enc].append((b, b.decode(enc)))
    return result


def _flatten_bytes(value):
    result = bytearray()
    for encoded, _ in value:
        result.extend(encoded)
    return bytes(result)


def _byte_exact_fraction(a, b):
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    same = 0
    for x, y in zip(a, b):
        if x == y:
            same += 1
    return same / len(a)


def grade(sol, fx) -> dict:
    cases = [
        ["hello", "café", "naïve"],
        ["A", "é", "ÿ", "Python"],
        ["Résumé", "123", "zoo"],
    ]

    reference_bytes = b""
    candidate_bytes = b""

    try:
        for strings in cases:
            ref = _oracle(strings)
            got = sol.encode_decode_round_trip(strings)

            for enc in ["utf-8", "utf-16-le", "latin-1"]:
                reference_bytes += _flatten_bytes(ref[enc])
                candidate_bytes += _flatten_bytes(got[enc])
    except Exception:
        return {"byte_exact_fraction": 0.0}

    return {
        "byte_exact_fraction": _byte_exact_fraction(
            reference_bytes, candidate_bytes
        )
    }
