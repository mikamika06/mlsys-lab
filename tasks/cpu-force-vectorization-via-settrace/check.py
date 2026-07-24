from mlsys.sim import cache as cachesim
import sys

def grade(sol, fx) -> dict:
    n = 64
    total = n * n

    # line event counter
    class LineCounter:
        def __init__(self):
            self.count = 0
            self.tracking = False
        def __call__(self, frame, event, arg):
            if event == 'line' and self.tracking:
                self.count += 1
            return self

    lc = LineCounter()
    sys.settrace(lc)
    lc.tracking = True
    try:
        order = sol.access_pattern(n)
    except Exception:
        lc.tracking = False
        sys.settrace(None)
        return {"covers_all": 0.0, "line_count": 10**6, "misses": 10**9}
    lc.tracking = False
    sys.settrace(None)

    line_count = lc.count

    if not isinstance(order, list):
        return {"covers_all": 0.0, "line_count": line_count, "misses": 10**9}

    # verify that it is a permutation
    if sorted(order) == list(range(total)):
        covers = 1.0
    else:
        covers = 0.0

    # build byte-address trace (float64 → 8 B per element)
    addrs = [i * 8 for i in order]
    sim = cachesim.simulate(addrs, line_bytes=64, sets=64, ways=8)
    misses = sim["misses"]

    return {"covers_all": covers, "line_count": line_count, "misses": misses}
