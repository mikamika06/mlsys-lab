def _ref(trace):
    current = 0
    max_allocated = 0
    reserved = 0
    alloc_sizes = {}
    next_id = 0
    for op, val in trace:
        if op == "alloc":
            size = val
            reserved += size
            current += size
            alloc_sizes[next_id] = size
            if current > max_allocated:
                max_allocated = current
            next_id += 1
        elif op == "free":
            id_ = val
            size = alloc_sizes[id_]
            current -= size
        else:
            raise ValueError(f"Unknown operation {op}")
    return reserved - max_allocated

def grade(sol, fx) -> dict:
    cases = [
        [("alloc", 10), ("alloc", 20), ("free", 0), ("free", 1)],
        [("alloc", 5), ("alloc", 15), ("free", 0), ("alloc", 10),
         ("free", 1), ("free", 2)],
        [("alloc", 100), ("free", 0)],
        [("alloc", 7), ("alloc", 3), ("free", 0), ("alloc", 4),
         ("free", 1), ("free", 2)],
    ]
    ok = 1.0
    for trace in cases:
        try:
            got = sol.fragmentation_gap(trace)
        except Exception:
            ok = 0.0
            break
        ref = _ref(trace)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
