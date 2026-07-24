import inspect


def _oracle_countdown(n):
    class Machine:
        def __init__(self, start):
            self.state = start

        def __next__(self):
            self.state -= 1
            if self.state < 0:
                raise StopIteration
            return self.state

    return Machine(n)


def _collect(it):
    out = []
    while True:
        try:
            out.append(next(it))
        except StopIteration:
            return out


def grade(sol, fx) -> dict:
    ok = 1.0

    try:
        source = inspect.getsource(sol.countdown)
        if "yield" in source:
            return {"exact_match": 0.0}
    except Exception:
        return {"exact_match": 0.0}

    for n in [0, 1, 2, 5, 10]:
        try:
            expected = _collect(_oracle_countdown(n))
            got_it = sol.countdown(n)

            if not hasattr(got_it, "state"):
                ok = 0.0
                break
            if not isinstance(got_it.state, int):
                ok = 0.0
                break
            if not hasattr(got_it, "__next__"):
                ok = 0.0
                break

            got = _collect(got_it)
            if got != expected:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break

    return {"exact_match": ok}
