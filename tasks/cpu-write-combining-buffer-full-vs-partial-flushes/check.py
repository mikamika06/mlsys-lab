from mlsys.sim import cache as cachesim


def _ref_wc(addrs, line_bytes, slots):
    entries = {}
    order = []
    full_flush = 0
    partial_flush = 0

    def flush(line):
        nonlocal full_flush, partial_flush
        offsets = entries[line]
        if len(offsets) == line_bytes:
            full_flush += 1
        else:
            partial_flush += 1
        del entries[line]
        order.remove(line)

    for addr in addrs:
        line = addr // line_bytes
        offset = addr % line_bytes
        if line not in entries:
            if len(entries) == slots:
                flush(order[0])
            entries[line] = set()
            order.append(line)
        entries[line].add(offset)
        if len(entries[line]) == line_bytes:
            flush(line)

    for line in list(order):
        flush(line)

    return full_flush, partial_flush


def _reference(addrs, line_bytes, slots):
    expected = _ref_wc(addrs, line_bytes, slots)
    cache = cachesim.simulate(
        addrs,
        line_bytes=64,
        sets=8,
        ways=2,
    )
    return expected, cache["misses"]


def grade(sol, fx) -> dict:
    cases = [
        (list(range(64)), 64, 2),
        (list(range(0, 128, 2)), 64, 2),
        ([0, 64, 128, 192, 256, 320], 64, 3),
        ([0, 4, 8, 12, 16, 20, 64, 68, 72], 32, 1),
        (list(range(0, 512, 8)), 64, 4),
    ]

    ok = 1.0
    for addrs, line_bytes, slots in cases:
        try:
            got = sol.wc_flush_stats(list(addrs), line_bytes, slots)
            ref, _ = _reference(list(addrs), line_bytes, slots)
        except Exception:
            ok = 0.0
            break
        if tuple(got) != tuple(ref):
            ok = 0.0
            break

    return {"exact_match": ok}
