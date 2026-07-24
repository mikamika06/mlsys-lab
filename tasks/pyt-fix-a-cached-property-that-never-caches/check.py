import sys


def _oracle_cached_property(func):
    name = func.__name__

    class Descriptor:
        def __get__(self, obj, owner=None):
            if obj is None:
                return self
            value = func(obj)
            obj.__dict__[name] = value
            return value

    return Descriptor()


def _oracle_compute_calls():
    class Sample:
        calls = 0

        @_oracle_cached_property
        def value(self):
            Sample.calls += 1
            return 17

    x = Sample()
    results = [x.value for _ in range(10)]
    return Sample.calls, results


def _count_lines(fn):
    count = 0

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line":
            count += 1
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        fn()
    finally:
        sys.settrace(old)
    return count


def grade(sol, fx) -> dict:
    oracle_calls, oracle_values = _oracle_compute_calls()

    try:
        class Sample:
            calls = 0

            @sol.cached_property
            def value(self):
                Sample.calls += 1
                return 17

        x = Sample()
        values = [x.value for _ in range(10)]
        calls = Sample.calls

        line_events = _count_lines(lambda: [x.value for _ in range(50)])
    except Exception:
        return {
            "compute_calls": 0.0,
            "line_events": 9999.0,
        }

    return {
        "compute_calls": 1.0 if calls == oracle_calls and values == oracle_values else 0.0,
        "line_events": float(line_events),
    }
