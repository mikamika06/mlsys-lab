def _oracle(names):
    def make_exception(name):
        if name == "ValueError":
            return ValueError()
        if name == "KeyError":
            return KeyError()
        if name == "IndexError":
            return IndexError()
        if name == "TypeError":
            return TypeError()
        if name == "RuntimeError":
            return RuntimeError()
        raise ValueError("unsupported fixture")

    fired = []
    group = ExceptionGroup("test", [make_exception(x) for x in names])
    try:
        raise group
    except* ValueError:
        fired.append(0)
    except* LookupError:
        fired.append(1)
    except* TypeError:
        fired.append(2)
    except* Exception:
        fired.append(3)
    return fired


def grade(sol, fx) -> dict:
    cases = [
        ["ValueError"],
        ["KeyError", "TypeError", "RuntimeError"],
        ["IndexError", "ValueError", "KeyError", "TypeError"],
        ["RuntimeError", "ValueError", "RuntimeError"],
        ["TypeError", "IndexError", "KeyError"],
    ]
    ok = 1.0
    for names in cases:
        try:
            expected = _oracle(names)
            got = sol.predict_except_star(list(names))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
