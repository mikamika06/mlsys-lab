def find_leaked_blocks(events: list[dict]) -> set[int]:
    allocs = {}
    for ev in events:
        op = ev["op"]
        sid = ev["seq_id"]
        if op == "alloc":
            if sid not in allocs:
                allocs[sid] = set()
            allocs[sid].update(ev["blocks"])
        elif op == "free":
            if sid in allocs:
                del allocs[sid]
        elif op == "free_block":
            if sid in allocs:
                allocs[sid].discard(ev["block"])

    leaked = set()
    for s in allocs.values():
        leaked.update(s)
    return leaked
