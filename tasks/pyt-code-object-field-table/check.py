def _oracle(fn):
    """The real oracle: read the fields straight off the live code object.
    There is no separate 'formula' for these six numbers — this IS what
    they mean — so the reference and the field descriptions below use the
    exact same CPython introspection API the learner is asked to use."""
    c = fn.__code__
    return (
        c.co_argcount,
        c.co_nlocals,
        c.co_stacksize,
        c.co_flags,
        len(c.co_consts),
        len(c.co_names),
    )


def _make_fixtures():
    def f_plain(a, b, c=3):
        x = a + b + c
        y = x * 2
        return y

    def f_varargs(*args, **kwargs):
        total = 0
        for a in args:
            total += a
        return total

    def f_generator(n):
        i = 0
        while i < n:
            yield i
            i += 1

    def f_closure(a, b):
        def inner(c):
            return a + b + c
        return inner

    class Box:
        def method(self, x, y=10):
            z = x * y
            return z

    return [f_plain, f_varargs, f_generator, f_closure, Box().method]


def grade(sol, fx) -> dict:
    fixtures = _make_fixtures()
    ok = 1.0
    for fn in fixtures:
        expected = _oracle(fn)
        try:
            got = tuple(sol.code_fields(fn))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
