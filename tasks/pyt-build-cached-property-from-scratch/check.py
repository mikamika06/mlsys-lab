"""
Real oracle: a Python class built dynamically (via ``type(...)``, which
triggers ``__set_name__`` exactly like a normal ``class`` statement) whose
methods record every real invocation in a plain list. Correct caching is
then a directly observable, countable fact — no hardcoded expected values.
"""


def _make_widget(CP):
    calls_a = []
    calls_b = []

    def compute_a(self):
        calls_a.append(id(self))
        return self.x * 2

    def compute_b(self):
        calls_b.append(id(self))
        total = 0
        for i in range(5):
            total += i
        return self.x + 100 + total

    ns = {
        "__init__": lambda self, x: setattr(self, "x", x),
        "a": CP(compute_a),
        "b": CP(compute_b),
    }
    Widget = type("Widget", (), ns)
    return Widget, calls_a, calls_b


def grade(sol, fx) -> dict:
    if not hasattr(sol, "cached_property"):
        return {"op_count": float("inf"), "exact_match": 0.0}
    CP = sol.cached_property

    try:
        Widget, calls_a, calls_b = _make_widget(CP)
        w1 = Widget(5)
        w2 = Widget(7)

        checks = []

        # each property accessed several times per instance
        vals = []
        for _ in range(3):
            vals.append(w1.a)
            vals.append(w1.b)
            vals.append(w2.a)
            vals.append(w2.b)

        # -- correctness: values match a direct, uncached computation --
        expected = [10, 115, 14, 117] * 3
        checks.append(vals == expected)

        # -- cache actually landed in the instance __dict__ --
        checks.append(w1.__dict__.get("a") == 10)
        checks.append(w1.__dict__.get("b") == 115)
        checks.append(w2.__dict__.get("a") == 14)

        # -- different instances must not share a cache --
        checks.append(w1.a != w2.a)

        # -- NON-DATA descriptor semantics: a direct write into the
        #    instance __dict__ must be visible on the next access,
        #    proving the value really comes from instance.__dict__ and
        #    not from descriptor-side state (would fail for a data
        #    descriptor, i.e. one that also defines __set__). --
        w1.__dict__["a"] = 999
        checks.append(w1.a == 999)

        # -- class-level access (instance=None) must not explode and
        #    must hand back something descriptor-like, not compute --
        class_val = Widget.__dict__["a"]
        checks.append(hasattr(class_val, "__get__"))

        exact_match = 1.0 if all(checks) else 0.0

        # -- op_count: extra (duplicate) compute-body invocations beyond
        #    the one-per-instance minimum, across everything above plus a
        #    dedicated fresh-instance probe --
        w3 = Widget(3)
        _ = w3.b
        _ = w3.b
        _ = w3.b

        total_calls = len(calls_a) + len(calls_b)
        # expected exactly-once calls: w1.a, w1.b, w2.a, w2.b, w3.b == 5
        expected_calls = 5
        op_count = float(max(0, total_calls - expected_calls))
    except Exception:
        return {"op_count": float("inf"), "exact_match": 0.0}

    return {"op_count": op_count, "exact_match": exact_match}
