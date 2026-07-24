def _oracle(strings):
    result = []
    for s in strings:
        result.append(
            (
                len(s),
                len(s.encode("utf-8")),
                len(s.encode("utf-16-le")),
            )
        )
    return result


def grade(sol, fx) -> dict:
    cases = [
        ["hello", "python"],
        ["café", "Ω", "Ж"],
        ["😀", "🦀", "𐍈"],
        ["a😀é𐍈"],
        ["", "A", "\u0000"],
    ]

    ok = 1.0
    for strings in cases:
        try:
            got = sol.encoding_lengths(list(strings))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(strings):
            ok = 0.0
            break

    return {"exact_match": ok}
