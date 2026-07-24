"""Grade collect_yield_from against the CPython generator state machine."""

# ---------------------------------------------------------------------------
# Reference oracle — drives a generator using next() + StopIteration.value
# ---------------------------------------------------------------------------

def _drive(gen):
    """Drive *gen* to exhaustion; return (values_list, return_value)."""
    values = []
    try:
        while True:
            values.append(next(gen))
    except StopIteration as exc:
        return values, exc.value

# ---------------------------------------------------------------------------
# Test generator factories
# ---------------------------------------------------------------------------

def _gen_plain():
    """No yield from — just plain yields."""
    yield 1
    yield 2
    yield 3

def _gen_from_no_return():
    """yield from a sub-iterator that has no explicit return."""
    def sub():
        yield 10
        yield 20
    yield from sub()
    yield 30

def _gen_from_with_return():
    """yield from with return-value capture, re-yielded."""
    def sub():
        yield 10
        yield 20
        return 'done'
    result = yield from sub()
    yield result

def _gen_nested():
    """Nested yield from — two delegation levels, returns at each."""
    def inner():
        yield 'a'
        return 1
    def outer():
        yield 'b'
        yield from inner()
        return 2
    result = yield from outer()
    yield result

def _gen_chained():
    """Sequential yield from calls, each capturing a return value."""
    def step1():
        yield 100
        return 'x'
    def step2():
        v = yield from step1()
        yield v
        return 'y'
    v = yield from step2()
    yield v

# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade(sol, fx) -> dict:
    factories = [
        _gen_plain,
        _gen_from_no_return,
        _gen_from_with_return,
        _gen_nested,
        _gen_chained,
    ]

    ok = 1.0
    for factory in factories:
        try:
            got = sol.collect_yield_from(factory())
        except Exception:
            ok = 0.0
            break
        try:
            expected = _drive(factory())
        except Exception:
            # Reference itself broke — should never happen
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
