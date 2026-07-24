import sys


def _reference(n, k):
    for i in range(n):
        value = i * i
        yield value


def _consume_k(it, k):
    out = []
    for _ in range(k):
        out.append(next(it))
    return out


def _trace_generator_events(it, k):
    if not hasattr(it, "gi_code"):
        return None

    code = it.gi_code
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line" and frame.f_code is code:
            count += 1
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        _consume_k(it, k)
    except Exception:
        return None
    finally:
        sys.settrace(old)
    return count


def grade(sol, fx) -> dict:
    cases = [
        (10, 3),
        (100, 1),
        (1000, 7),
        (20, 20),
    ]

    exact = 1.0
    budget = 1.0

    for n, k in cases:
        expected = _consume_k(_reference(n, k), k)
        try:
            got_it = sol.take_k_squares(n, k)
            got = _consume_k(got_it, k)
        except Exception:
            exact = 0.0
            budget = 0.0
            break

        if got != expected:
            exact = 0.0

        ref_events = _trace_generator_events(_reference(n, k), k)
        got_it = sol.take_k_squares(n, k)
        got_events = _trace_generator_events(got_it, k)

        if got_events is None or ref_events is None or got_events != ref_events:
            budget = 0.0

    return {
        "exact_match": exact,
        "lazy_line_budget": budget,
    }
