def _clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))


def grade(sol, fx) -> dict:
    try:
        Widget = sol.Widget
        priv = Widget.level.private_name  # e.g. "_level" — read off the
        # descriptor itself, whatever private name the (unmodified)
        # __init__ chose, rather than hardcoding "_level".

        results = []

        # 1) in-range and out-of-range values clamp correctly on construction
        for v in (42, 500, -30, 0, 100):
            w = Widget(v)
            results.append(w.level == _clamp(v))
            # a proper data descriptor never leaves a same-named public
            # key sitting in the instance __dict__
            results.append("level" not in w.__dict__)
            results.append(w.__dict__.get(priv) == _clamp(v))

        # 2) two instances stay independent (no shared/class-level state)
        w1 = Widget(10)
        w2 = Widget(90)
        results.append(w1.level == 10 and w2.level == 90)

        # 3) reassignment after construction clamps too
        w = Widget(0)
        w.level = 75
        results.append(w.level == 75)
        w.level = 999
        results.append(w.level == 100)

        # 4) the key regression check: directly poking the instance dict
        # under the PUBLIC name must NOT be able to shadow the descriptor.
        # Under the original bug (non-data descriptor) this poke would win
        # and w.level would read back 12345; a proper data descriptor
        # always wins the lookup regardless of what's sitting in __dict__.
        w = Widget(20)
        w.__dict__["level"] = 12345
        results.append(w.level == 20)

        # normal assignment afterwards still works, routed through the
        # private storage key rather than the poked public one
        w.level = 55
        results.append(w.level == 55)
        results.append(w.__dict__.get(priv) == 55)

        ok = 1.0 if all(results) else 0.0
    except Exception:
        ok = 0.0

    return {"exact_match": ok}
