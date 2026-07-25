import dis


def _uses_forbidden_encode(fn):
    for ins in dis.get_instructions(fn):
        if ins.opname in {"LOAD_METHOD", "LOAD_ATTR"} and ins.argval == "encode":
            return True
    return False


def _oracle(text):
    return text.encode("utf-8")


def grade(sol, fx) -> dict:
    cases = [
        "",
        "hello",
        "¢",
        "€",
        "你好",
        "😀",
        "A¢€😀",
        "mañana",
        "𐍈𝄞",
        "UTF-8 ✓ works",
    ]

    try:
        if _uses_forbidden_encode(sol.utf8_encode):
            return {"byte_exact_fraction": 0.0}
    except Exception:
        return {"byte_exact_fraction": 0.0}

    total = 0
    matched = 0
    for text in cases:
        expected = _oracle(text)
        # count the bytes we are asking for even when the solver blows up,
        # otherwise a solver that always raises measures nothing and a
        # "nothing measured" score would look like a perfect one
        total += len(expected)
        try:
            got = sol.utf8_encode(text)
        except Exception:
            continue
        if not isinstance(got, (bytes, bytearray)) or len(got) != len(expected):
            continue
        matched += sum(1 for a, b in zip(got, expected) if a == b)

    if total == 0:                      # every case is the empty string: no evidence
        return {"byte_exact_fraction": 0.0}
    return {"byte_exact_fraction": float(matched / total)}
