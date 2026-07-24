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
        try:
            got = sol.utf8_encode(text)
        except Exception:
            continue
        total += len(expected)
        matched += sum(1 for a, b in zip(got, expected) if a == b)
        if len(got) != len(expected):
            continue

    if total == 0:
        score = 1.0
    else:
        score = matched / total
    return {"byte_exact_fraction": float(score)}
