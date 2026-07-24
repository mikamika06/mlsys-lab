"""Oracle: independent re-implementation of the caching allocator, plus a
trace driver that replays malloc/free traces against both the oracle and the
student's CachingAllocator and compares the observed (success, reserved)
sequences. Block ids are opaque -- only success/OOM and reserved-byte totals
are checked, exactly as production caching-allocator tests do it.
"""
_BLOCK = 512


def _round(nbytes):
    if nbytes <= _BLOCK:
        return _BLOCK
    return _BLOCK * ((nbytes + _BLOCK - 1) // _BLOCK)


class _Oracle:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reserved = 0
        self._free = []  # list of {'tag', 'size'}, insertion order
        self._live = {}  # tag -> size
        self._n = 0

    def malloc(self, nbytes):
        size = _round(nbytes)
        best_i = None
        for i, blk in enumerate(self._free):
            if blk["size"] >= size:
                if best_i is None or blk["size"] < self._free[best_i]["size"]:
                    best_i = i
        if best_i is not None:
            blk = self._free.pop(best_i)
            remainder = blk["size"] - size
            if remainder > 0:
                self._free.append({"tag": self._tag(), "size": remainder})
            tag = self._tag()
            self._live[tag] = size
            return tag
        if self.reserved + size > self.capacity:
            return None
        self.reserved += size
        tag = self._tag()
        self._live[tag] = size
        return tag

    def free(self, tag):
        size = self._live.pop(tag)
        self._free.append({"tag": self._tag(), "size": size})

    def _tag(self):
        self._n += 1
        return self._n


def _run_oracle(capacity, trace):
    o = _Oracle(capacity)
    names = {}
    events = []
    for op in trace:
        if op[0] == "malloc":
            _, name, nbytes = op
            bid = o.malloc(nbytes)
            names[name] = bid
            events.append((bid is not None, o.reserved))
        else:
            _, name = op
            bid = names.pop(name, None)
            if bid is not None:
                o.free(bid)
            events.append((True, o.reserved))
    return events


def _run_sol(sol, capacity, trace):
    a = sol.CachingAllocator(capacity)
    names = {}
    events = []
    for op in trace:
        if op[0] == "malloc":
            _, name, nbytes = op
            bid = a.malloc(nbytes)
            names[name] = bid
            events.append((bid is not None, int(a.reserved)))
        else:
            _, name = op
            bid = names.pop(name, None)
            if bid is not None:
                a.free(bid)
            events.append((True, int(a.reserved)))
    return events


def _traces():
    return [
        (
            2048,
            [
                ("malloc", "a", 100),   # rounds to 512, miss -> reserved 512
                ("malloc", "b", 600),   # rounds to 1024, miss -> reserved 1536
                ("free", "a"),          # 512 back to free list
                ("malloc", "c", 400),   # rounds to 512, best-fit reuses freed a
                ("malloc", "d", 700),   # rounds to 1024, miss -> reserved 2560 > 2048 -> OOM
            ],
        ),
        (
            4096,
            [
                ("malloc", "a", 2000),  # rounds to 2048, miss -> reserved 2048
                ("free", "a"),
                ("malloc", "b", 512),   # best-fit reuses a's block (2048>=512), splits: 512 used, 1536 remainder free
                ("malloc", "c", 1000),  # rounds to 1024, best-fit reuses the 1536 remainder, splits again (512 left free)
                ("malloc", "d", 500),   # rounds to 512, best-fit reuses the 512 remainder exactly (no split)
                ("free", "b"),
                ("free", "c"),
                ("free", "d"),
                ("malloc", "e", 2048),  # exact best fit across the three freed same-size-ish blocks
            ],
        ),
        (
            1536,
            [
                ("malloc", "a", 512),
                ("malloc", "b", 512),
                ("malloc", "c", 512),   # reserved hits capacity exactly (1536)
                ("malloc", "d", 1),     # miss, no free blocks -> OOM
                ("free", "b"),
                ("malloc", "e", 512),   # best-fit reuses b's block
                ("malloc", "f", 1),     # miss again, capacity full -> OOM
            ],
        ),
    ]


def grade(sol, fx) -> dict:
    ok = 1.0
    for capacity, trace in _traces():
        expected = _run_oracle(capacity, trace)
        try:
            got = _run_sol(sol, capacity, trace)
        except Exception:
            return {"exact_match": 0.0}
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
