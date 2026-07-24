import dis

EXPRS = [
    "1 + 2",
    "2 ** 10",
    "10 / 4",
    "1 / 0",
    "'a' + 'b'",
    "'ab' * 3",
    "(1, 2, 3)",
    "[1, 2, 3]",
    "{1: 2}",
    "{1, 2, 3}",
    "True and False",
    "2 ** 1000",
    "-5",
    "1 < 2",
    "x + 1",
]


def _oracle(expr):
    """Compile the expression as a lambda body and disassemble it — the
    real, live CPython compiler is the only source of truth here."""
    code = compile("lambda: " + expr, "<oracle>", "eval")
    fn = eval(code)
    ops = [ins.opname for ins in dis.get_instructions(fn) if ins.opname != "RESUME"]
    return ops == ["LOAD_CONST", "RETURN_VALUE"] or ops == ["RETURN_CONST"]


def grade(sol, fx) -> dict:
    expected = [_oracle(e) for e in EXPRS]
    try:
        got = list(sol.predict_folded(list(EXPRS)))
    except Exception:
        return {"exact_match": 0.0}

    if len(got) != len(expected):
        return {"exact_match": 0.0}

    ok = all(bool(g) == bool(e) for g, e in zip(got, expected))
    return {"exact_match": 1.0 if ok else 0.0}
