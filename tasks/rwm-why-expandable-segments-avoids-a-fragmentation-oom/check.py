def _oracle_replay(trace, capacity, expandable=False):
    segments = {}      # seg_id -> {"free_starts": {}, "free_ends": {}}
    live = {}            # name -> (seg_id, start, size)
    reserved = 0
    peak = 0
    oom = False
    next_seg = [0]

    def new_segment():
        sid = next_seg[0]
        next_seg[0] += 1
        segments[sid] = {"free_starts": {}, "free_ends": {}}
        return sid

    def add_free(sid, start, size):
        seg = segments[sid]
        fs, fe = seg["free_starts"], seg["free_ends"]
        end = start + size
        if end in fs:
            rsize = fs.pop(end)
            rend = end + rsize
            del fe[rend]
            size += rsize
            end = start + size
        if start in fe:
            lstart = fe.pop(start)
            lsize = fs.pop(lstart)
            size += lsize
            start = lstart
            end = start + size
        fs[start] = size
        fe[end] = start

    single_sid = new_segment() if expandable else None

    for op in trace:
        if oom:
            break
        if op[0] == "alloc":
            _, name, size = op
            cands = []
            for sid, seg in segments.items():
                for s, sz in seg["free_starts"].items():
                    if sz >= size:
                        cands.append((sz, sid, s))
            if cands:
                cands.sort()
                sz, sid, s = cands[0]
                seg = segments[sid]
                end = s + sz
                del seg["free_starts"][s]
                del seg["free_ends"][end]
                if sz > size:
                    add_free(sid, s + size, sz - size)
                live[name] = (sid, s, size)
                peak = max(peak, reserved)
            elif expandable:
                if reserved + size > capacity:
                    oom = True
                    continue
                start = reserved
                reserved += size
                live[name] = (single_sid, start, size)
                peak = max(peak, reserved)
            else:
                if reserved + size > capacity:
                    oom = True
                    continue
                sid = new_segment()
                reserved += size
                live[name] = (sid, 0, size)
                peak = max(peak, reserved)
        else:
            _, name = op
            sid, s, sz = live.pop(name)
            add_free(sid, s, sz)

    return {"oom": oom, "peak_reserved": peak}


def _cases():
    trace_a = [
        ("alloc", "A", 700),
        ("alloc", "B", 700),
        ("free", "A"),
        ("free", "B"),
        ("alloc", "C", 1400),
    ]
    trace_b = [
        ("alloc", "x", 200),
        ("alloc", "A", 500),
        ("alloc", "B", 500),
        ("alloc", "C", 500),
        ("free", "A"),
        ("free", "B"),
        ("free", "C"),
        ("alloc", "D", 1500),
    ]
    trace_c = [
        ("alloc", "A", 900),
        ("alloc", "B", 300),
        ("free", "A"),
        ("alloc", "C", 300),
        ("free", "B"),
        ("free", "C"),
        ("alloc", "D", 1200),
    ]
    return [
        (trace_a, 2000),
        (trace_b, 2700),
        (trace_c, 1800),
    ]


def grade(sol, fx) -> dict:
    ok = 1.0
    for trace, capacity in _cases():
        for expandable in (False, True):
            expected = _oracle_replay(trace, capacity, expandable)
            try:
                got = sol.replay_trace(list(trace), capacity, expandable)
                got = {"oom": bool(got["oom"]), "peak_reserved": int(got["peak_reserved"])}
            except Exception:
                return {"exact_match": 0.0}
            if got != expected:
                ok = 0.0
                break
        if ok == 0.0:
            break
    return {"exact_match": ok}
