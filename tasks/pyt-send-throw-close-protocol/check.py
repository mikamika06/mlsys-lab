def _oracle_generator():
    try:
        value = yield "ready"
        while True:
            try:
                value = yield "received:" + value
            except ValueError as exc:
                value = yield "handled:" + str(exc)
    finally:
        return "closed"


def _drive(factory):
    g = factory()
    out = []
    try:
        out.append(("next", next(g)))
    except Exception as exc:
        out.append(("next_exc", type(exc).__name__))
        return out

    try:
        out.append(("send", g.send("alpha")))
    except Exception as exc:
        out.append(("send_exc", type(exc).__name__))

    try:
        out.append(("throw", g.throw(ValueError("bad"))))
    except Exception as exc:
        out.append(("throw_exc", type(exc).__name__))

    try:
        out.append(("send", g.send("beta")))
    except Exception as exc:
        out.append(("send_exc", type(exc).__name__))

    try:
        g.close()
        out.append(("close", "closed"))
    except Exception as exc:
        out.append(("close_exc", type(exc).__name__))
    return out


def _drive_real_oracle():
    g = _oracle_generator()
    out = []
    out.append(("next", next(g)))
    out.append(("send", g.send("alpha")))
    out.append(("throw", g.throw(ValueError("bad"))))
    out.append(("send", g.send("beta")))
    try:
        g.close()
        out.append(("close", "closed"))
    except StopIteration as exc:
        out.append(("close", exc.value))
    return out


def grade(sol, fx) -> dict:
    expected = _drive_real_oracle()
    try:
        got = sol.run_protocol()
    except Exception:
        return {"exact_match": 0.0}
    return {"exact_match": 1.0 if got == expected else 0.0}
