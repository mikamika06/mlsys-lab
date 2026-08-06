def audit_block_trace(trace):
    allocated = set()
    active_per_seq = {}
    double_free = 0
    use_after_free = 0
    leaked = 0

    for event in trace:
        etype = event["type"]
        seq_id = event.get("seq_id")
        block_id = event.get("block_id")

        if etype == "allocate":
            if block_id in allocated:
                double_free += 1
            allocated.add(block_id)
            active_per_seq.setdefault(seq_id, set()).add(block_id)
        elif etype == "free":
            if block_id not in allocated:
                use_after_free += 1
            else:
                allocated.remove(block_id)
                if seq_id in active_per_seq and block_id in active_per_seq[seq_id]:
                    active_per_seq[seq_id].remove(block_id)
        elif etype == "terminate":
            if seq_id in active_per_seq:
                leaked += len(active_per_seq[seq_id])
                for b in active_per_seq[seq_id]:
                    if b in allocated:
                        allocated.remove(b)
                del active_per_seq[seq_id]

    leaked += len(allocated)
    return {
        "double_free": double_free,
        "use_after_free": use_after_free,
        "leaked": leaked,
    }
