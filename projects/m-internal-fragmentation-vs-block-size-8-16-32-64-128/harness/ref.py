def compute_fragmentation(seq_lens, block_sizes):
    res = {}
    for bs in block_sizes:
        frag = sum(((l + bs - 1) // bs) * bs - l for l in seq_lens)
        res[bs] = frag
    return res

def gather_slot_mapping(seq_lens, block_tables, block_size):
    slots = []
    for l, table in zip(seq_lens, block_tables):
        for i in range(l):
            slots.append(table[i // block_size] * block_size + (i % block_size))
    return slots

def find_leaked_blocks(events):
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

def gen_trace():
    return [
        {"op": "alloc", "seq_id": 1, "blocks": [10, 11]},
        {"op": "alloc", "seq_id": 2, "blocks": [12]},
        {"op": "alloc", "seq_id": 1, "blocks": [13]},
        {"op": "free_block", "seq_id": 1, "block": 10},
        {"op": "free", "seq_id": 2},
        {"op": "alloc", "seq_id": 3, "blocks": [14, 15]},
    ]
