import ref

def check(workdir):
    from pagedkv.allocator import BlockAllocator, SequenceManager, compute_slot_mapping
    out = {"cow_match": 0.0, "slot_mapping_match": 0.0}
    try:
        wt1, wt2, wsm, wrc = ref.oracle_cow_and_slot(10, 4)
        alloc = BlockAllocator(10, 4)
        mgr = SequenceManager(alloc, 4)
        mgr.create_sequence(1, 3)
        mgr.fork(1, 2)
        mgr.append_token(2)
        gt1 = mgr.get_block_table(1)
        gt2 = mgr.get_block_table(2)
        gsm = compute_slot_mapping([gt1, gt2], [3, 4], 4)
        if gt1 == wt1 and gt2 == wt2:
            out["cow_match"] = 1.0
        if gsm == wsm:
            out["slot_mapping_match"] = 1.0
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:100]}"
    return out
