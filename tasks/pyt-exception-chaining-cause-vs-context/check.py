def _oracle(mode):
    try:
        try:
            raise ValueError("inner")
        except ValueError as exc:
            if mode == "from":
                raise RuntimeError("outer") from exc
            if mode == "context":
                raise RuntimeError("outer")
            raise ValueError("bad mode")
    except Exception as err:
        cause = err.__cause__
        context = err.__context__
        return {
            "cause": cause is not None,
            "context": context is not None,
            "cause_type": type(cause).__name__ if cause is not None else None,
            "context_type": type(context).__name__ if context is not None else None,
        }


def grade(sol, fx) -> dict:
    expected = {
        mode: _oracle(mode)
        for mode in ("from", "context")
    }
    ok = 1.0
    for mode, ref in expected.items():
        try:
            got = sol.inspect_exception_chain(mode)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
