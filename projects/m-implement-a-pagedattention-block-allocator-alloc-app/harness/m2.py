import ref

def check(workdir):
    try:
        from kv.allocator import PagedAllocator
        oracle = ref.OracleAllocator(30, 2)
        learner = PagedAllocator(30, 2)
        ops = ref.get_m2_ops()
        match = 0
        free_match = 0
        for op, seq, arg in ops:
            if op == "alloc":
                oracle.alloc(seq, arg)
                learner.alloc(seq, arg)
            elif op == "append":
                oracle.append(seq, arg)
                learner.append(seq, arg)
            elif op == "fork":
                oracle.fork(seq, arg)
                learner.fork(seq, arg)
            elif op == "free":
                oracle.free(seq)
                learner.free(seq)

            chk_seq = arg if op == "fork" else seq
            if oracle.reconstruct(chk_seq) == learner.reconstruct(chk_seq):
                match += 1

            if len(oracle.free_blocks) == len(learner.free_blocks):
                free_match += 1

        out = {}
        out["logic_match"] = 1.0 if match == len(ops) else 0.0
        out["memory_match"] = 1.0 if free_match == len(ops) else 0.0
        return out
    except Exception as e:
        return {"logic_match": 0.0, "memory_match": 0.0, "_note": f"{type(e).__name__}: {e}"}
