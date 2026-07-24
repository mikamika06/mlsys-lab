def _shape(exc):
    if exc is None:
        return None
    if isinstance(exc, BaseExceptionGroup):
        return (
            "group",
            exc.message,
            tuple(_shape(item) for item in exc.exceptions),
        )
    return ("leaf", type(exc).__name__)


def _oracle(eg, names):
    return eg.split(lambda e: type(e).__name__ in names)


def grade(sol, fx) -> dict:
    cases = [
        (
            ExceptionGroup(
                "root",
                [
                    ValueError("bad"),
                    ExceptionGroup(
                        "nested",
                        [TypeError("wrong"), KeyError("missing")],
                    ),
                ],
            ),
            ("ValueError", "KeyError"),
        ),
        (
            ExceptionGroup(
                "outer",
                [
                    ExceptionGroup(
                        "left",
                        [IndexError("x"), ValueError("y")],
                    ),
                    RuntimeError("z"),
                    ExceptionGroup(
                        "right",
                        [LookupError("a")],
                    ),
                ],
            ),
            ("IndexError",),
        ),
        (
            ExceptionGroup(
                "all",
                [
                    OSError("disk"),
                    ExceptionGroup(
                        "deep",
                        [
                            ExceptionGroup(
                                "deeper",
                                [ValueError("v")],
                            )
                        ],
                    ),
                ],
            ),
            ("ValueError", "OSError"),
        ),
        (
            ExceptionGroup(
                "none",
                [
                    TypeError("a"),
                    ExceptionGroup("child", [RuntimeError("b")]),
                ],
            ),
            ("KeyError",),
        ),
    ]

    ok = 1.0
    for eg, names in cases:
        try:
            ref = _oracle(eg, names)
            got = sol.split_group(eg, names)
            ref_shape = (_shape(ref[0]), _shape(ref[1]))
            got_shape = (_shape(got[0]), _shape(got[1]))
        except Exception:
            ok = 0.0
            break
        if got_shape != ref_shape:
            ok = 0.0
            break
    return {"exact_match": ok}
