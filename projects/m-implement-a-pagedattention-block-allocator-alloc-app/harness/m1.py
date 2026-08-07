import ref

def check(workdir):
    try:
        from kv.allocator import PagedAllocator
        oracle = ref.OracleAllocator(20, 4)
        learner = PagedAllocator(20, 4)
        ops = ref.get_m1_ops()
        match = 0
        for op, seq, arg in ops:
            if op == "alloc":
                oracle.alloc(seq, arg)
                learner.alloc(seq, arg)
            elif op == "append":
                oracle.append(seq, arg)
                learner.append(seq, arg)
            elif op == "free":
                oracle.free(seq)
                learner.free(seq)

            if oracle.reconstruct(seq) == learner.reconstruct(seq):
                match += 1

        return {"all_match": 1.0 if match == len(ops) else 0.0}
    except Exception as e:
        return {"all_match": 0.0, "_note": f"{type(e).__name__}: {e}"}
