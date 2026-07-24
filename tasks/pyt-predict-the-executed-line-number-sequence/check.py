import sys


def _trace_oracle(fn):
    target = fn.__code__
    lines = []

    def tracer(frame, event, arg):
        if frame.f_code is target and event == "line":
            lines.append(frame.f_lineno)
        return tracer

    sys.settrace(tracer)
    try:
        fn()
    finally:
        sys.settrace(None)
    return lines


def _fixture_a():
    total = 0
    for i in range(3):
        total += i
    return total


def _fixture_b():
    x = 2
    if x > 1:
        y = x + 4
    else:
        y = x - 4
    return y


def _fixture_c():
    value = 0
    for i in range(2):
        if i == 0:
            value += 10
        else:
            value += 20
    return value


def grade(sol, fx) -> dict:
    cases = [_fixture_a, _fixture_b, _fixture_c]
    ok = 1.0
    for fn in cases:
        try:
            got = sol.predict_line_sequence(fn)
        except Exception:
            ok = 0.0
            break
        if got != _trace_oracle(fn):
            ok = 0.0
            break
    return {"exact_match": ok}
