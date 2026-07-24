def _oracle_replay(trace, capacity, max_split_size=None):
    free_starts = {}   # start -> size
    free_ends = {}      # end -> start
    live = {}            # name -> (start, size)
    reserved = 0
    peak = 0
    oom = False

    def add_free(start, size):
        end = start + size
        if end in free_starts:
            rsize = free_starts.pop(end)
            rend = end + rsize
            del free_ends[rend]
            size += rsize
            end = start + size
        if start in free_ends:
            lstart = free_ends.pop(start)
            lsize = free_starts.pop(lstart)
            size += lsize
            start = lstart
            end = start + size
        free_starts[start] = size
        free_ends[end] = start

    for op in trace:
        if oom:
            break
        if op[0] == "alloc":
            _, name, size = op
            cands = []
            for s, sz in free_starts.items():
                if sz >= size and (max_split_size is None or sz <= max_split_size or sz == size):
                    cands.append((sz, s))
            if cands:
                cands.sort()
                sz, s = cands[0]
                end = s + sz
                del free_starts[s]
                del free_ends[end]
                if sz > size:
                    add_free(s + size, sz - size)
                live[name] = (s, size)
                peak = max(peak, reserved)
            else:
                if reserved + size > capacity:
                    oom = True
                    continue
                start = reserved
                reserved += size
                live[name] = (start, size)
                peak = max(peak, reserved)
        else:
            _, name = op
            s, sz = live.pop(name)
            add_free(s, sz)

    return {"oom": oom, "peak_reserved": peak}


def _cases():
    trace_a = [
        ("alloc", "A", 1000),
        ("free", "A"),
        ("alloc", "s1", 100),
        ("alloc", "s2", 100),
        ("alloc", "s3", 100),
        ("alloc", "s4", 100),
        ("alloc", "B", 1000),
    ]
    trace_b = [
        ("alloc", "A", 2000),
        ("free", "A"),
        ("alloc", "s1", 200),
        ("alloc", "s2", 200),
        ("alloc", "s3", 200),
        ("free", "s2"),
        ("alloc", "B", 2000),
    ]
    trace_c = [
        ("alloc", "x", 300),
        ("alloc", "A", 1200),
        ("free", "A"),
        ("alloc", "s1", 100),
        ("alloc", "s2", 100),
        ("alloc", "s3", 100),
        ("alloc", "B", 1200),
    ]
    return [
        (trace_a, 1500, 500),
        (trace_b, 2600, 800),
        (trace_c, 2000, 400),
    ]


def grade(sol, fx) -> dict:
    ok = 1.0
    for trace, capacity, cap in _cases():
        for msc in (None, cap):
            expected = _oracle_replay(trace, capacity, msc)
            try:
                got = sol.replay_trace(list(trace), capacity, msc)
                got = {"oom": bool(got["oom"]), "peak_reserved": int(got["peak_reserved"])}
            except Exception:
                return {"exact_match": 0.0}
            if got != expected:
                ok = 0.0
                break
        if ok == 0.0:
            break
    return {"exact_match": ok}
