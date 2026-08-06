import ref

def check(workdir):
    from pagedkv.allocator import BlockAllocator
    out = {"alloc_free_match": 0.0}
    ops = [("alloc", None), ("alloc", None), ("free", 0), ("alloc", None)]
    try:
        want = ref.oracle_alloc_free(10, 4, ops)
        alloc = BlockAllocator(10, 4)
        got = []
        for op, val in ops:
            if op == "alloc":
                got.append(alloc.alloc())
            elif op == "free":
                alloc.free(val)
                got.append(alloc.get_ref_count(val))
        if got == want:
            out["alloc_free_match"] = 1.0
        else:
            out["_note"] = f"got {got}, want {want}"
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)[:100]}"
    return out
